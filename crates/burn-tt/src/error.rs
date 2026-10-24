//! The backend's error type.
//!
//! Burn's ops return tensors, not results, so most of these reach a caller as
//! a panic message, or as an `ExecutionError` from `try_into_data` and `sync`.
//! The variants exist so the message can carry the fix, and so a test or a
//! caller can tell "no card" from "the card hung" without matching text.

use std::sync::Arc;

use tt_device::core_control::WaitError;
use tt_device::TransportError;
use tt_kernels::dm::DmError;
use tt_kernels::runtime::RunError;
use tt_kernels::session::SessionError;
use tt_kernels::tensor::TensorError;

use crate::TtDevice;

/// Why the backend could not open a card or finish a job.
///
/// `Clone` because the first failure of an attachment is remembered and
/// reported again by every later wait (`server::Ids`); the lower-level errors
/// are not `Clone`, so they are classified here and kept as text.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TtError {
    /// `/dev/tenstorrent` does not exist: the kernel driver is not loaded.
    NoDriver,
    /// This card has no device node.
    NoSuchCard { card: u16, present: Vec<u16> },
    /// The device node exists but this user cannot open it.
    Permission { path: String },
    /// Another process (or another session of this one) holds the card.
    CardBusy { card: u16 },
    /// The requested tile is fused off (or never existed) on this chip.
    NoSuchTile { x: u8, y: u8, columns: Vec<u8> },
    /// More tiles were asked for than the chip has.
    TooFewTiles { asked: usize, have: usize },
    /// A run or a data mover did not finish within its budget.
    Timeout { detail: String },
    /// No GDDR channel has room for a tensor: the channel that refused it had `free` bytes, the
    /// largest run of them `largest`.
    OutOfMemory { slots: u64, free: u64, largest: u64 },
    /// The engine cannot do this (an engine without device storage, a trace on
    /// an engine that captures none, an op the mesh does not run).
    Unsupported { what: String },
    /// Any other device or engine failure, with its text.
    Device(String),
    /// No engine serves this device and none could be started.
    NotAttached { device: TtDevice },
    /// The device is already attached; `auto` says by the registry rather than
    /// by an explicit [`attach`](crate::attach).
    AlreadyAttached { device: TtDevice, auto: bool },
    /// A tensor that outlived the attachment it was made in.
    StaleTensor { id: u64 },
    /// The device's server thread panicked while running a job.
    ServerPanicked { device: TtDevice, message: String },
    /// `cause`, while running `what` on `device`: how an asynchronous op's
    /// failure is recorded.
    Op {
        what: String,
        device: TtDevice,
        cause: Arc<TtError>,
    },
    /// A wait that found an earlier asynchronous op had failed.
    EarlierFailure(Arc<TtError>),
}

impl TtError {
    /// An error for something the engine does not do.
    pub fn unsupported(what: impl Into<String>) -> Self {
        TtError::Unsupported { what: what.into() }
    }

    /// The failure underneath any [`TtError::Op`] / [`TtError::EarlierFailure`]
    /// wrapping.
    pub fn root(&self) -> &TtError {
        match self {
            TtError::Op { cause, .. } | TtError::EarlierFailure(cause) => cause.root(),
            other => other,
        }
    }
}

impl std::fmt::Display for TtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtError::NoDriver => f.write_str(
                "/dev/tenstorrent does not exist: is the tenstorrent kernel driver loaded?",
            ),
            TtError::NoSuchCard { card, present } => write!(
                f,
                "no card {card} (/dev/tenstorrent/{card} does not exist); cards present: {present:?}; \
                 name one of them with burn_tt::init, or set TT_TOPOLOGY"
            ),
            TtError::Permission { path } => write!(
                f,
                "permission denied opening {path}: is the user in the group that owns it?"
            ),
            TtError::CardBusy { card } => write!(
                f,
                "card {card} is in use by another process (or another session in this one); \
                 close it first"
            ),
            TtError::NoSuchTile { x, y, columns } => write!(
                f,
                "tile ({x}, {y}) is fused off or does not exist on this chip; its tile columns \
                 are {columns:?}"
            ),
            TtError::TooFewTiles { asked, have } => {
                write!(f, "{asked} tiles asked for, the chip has {have}")
            }
            TtError::Timeout { detail } => write!(f, "{detail}"),
            TtError::OutOfMemory {
                slots,
                free,
                largest,
            } => write!(
                f,
                "no room in GDDR for {slots} tile slots per channel (the channel that refused \
                 them has {free} bytes free, the largest run {largest}): free tensors, or use a \
                 smaller batch"
            ),
            TtError::Unsupported { what } => f.write_str(what),
            TtError::Device(text) => f.write_str(text),
            TtError::NotAttached { device } => write!(
                f,
                "{device} is not attached and could not be: call burn_tt::init, or burn_tt::attach"
            ),
            TtError::AlreadyAttached { device, auto: false } => {
                write!(f, "{device} is already attached")
            }
            TtError::AlreadyAttached { device, auto: true } => write!(
                f,
                "{device} is already attached (it was attached on first use): call \
                 burn_tt::detach first, or configure it with burn_tt::init before any op"
            ),
            TtError::StaleTensor { id } => write!(
                f,
                "buffer {id} is not one of this attachment's: a tensor that outlived the \
                 attachment it was made in"
            ),
            TtError::ServerPanicked { device, message } => {
                write!(f, "{device}'s server thread panicked: {message}")
            }
            TtError::Op {
                what,
                device,
                cause,
            } => write!(f, "{what} on {device}: {cause}"),
            TtError::EarlierFailure(cause) => {
                write!(f, "an earlier device op failed: {cause}")
            }
        }
    }
}

impl std::error::Error for TtError {}

fn timed_out(e: &WaitError) -> bool {
    matches!(e, WaitError::TimedOut { .. })
}

impl From<WaitError> for TtError {
    fn from(e: WaitError) -> Self {
        if timed_out(&e) {
            TtError::Timeout {
                detail: e.to_string(),
            }
        } else {
            TtError::Device(e.to_string())
        }
    }
}

impl From<TransportError> for TtError {
    fn from(e: TransportError) -> Self {
        TtError::Device(e.to_string())
    }
}

impl From<DmError> for TtError {
    fn from(e: DmError) -> Self {
        match e {
            DmError::TimedOut { .. } => TtError::Timeout {
                detail: e.to_string(),
            },
            DmError::Transport(t) => t.into(),
            other => TtError::Device(other.to_string()),
        }
    }
}

impl From<RunError> for TtError {
    fn from(e: RunError) -> Self {
        let timeout = match &e {
            RunError::Setup(w) => timed_out(w),
            RunError::Roles(roles) => roles.iter().any(|(_, _, w)| timed_out(w)),
            _ => false,
        };
        match e {
            RunError::Transport(t) => t.into(),
            // A tile whose reset never finished is not a timeout: its text
            // carries the board-reset hint.
            other if timeout => TtError::Timeout {
                detail: other.to_string(),
            },
            other => TtError::Device(other.to_string()),
        }
    }
}

impl From<TensorError> for TtError {
    fn from(e: TensorError) -> Self {
        match e {
            TensorError::OutOfMemory {
                slots,
                free,
                largest,
            } => TtError::OutOfMemory {
                slots,
                free,
                largest,
            },
            TensorError::Transport(t) => t.into(),
            TensorError::Mover(m) => m.into(),
            TensorError::Run(r) => r.into(),
            other => TtError::Device(other.to_string()),
        }
    }
}

impl From<SessionError> for TtError {
    fn from(e: SessionError) -> Self {
        match e {
            SessionError::CardBusy { card } => TtError::CardBusy { card },
            SessionError::NoSuchTile { x, y, columns } => TtError::NoSuchTile { x, y, columns },
            SessionError::TooFewTiles { asked, have } => TtError::TooFewTiles { asked, have },
            SessionError::Transport(t) => t.into(),
            SessionError::Reset(r) => r.into(),
            other => TtError::Device(other.to_string()),
        }
    }
}

impl From<tt_kernels::mesh_trace::MeshTraceError> for TtError {
    fn from(e: tt_kernels::mesh_trace::MeshTraceError) -> Self {
        TtError::Device(e.to_string())
    }
}

impl From<tt_kernels::link::LinkError> for TtError {
    fn from(e: tt_kernels::link::LinkError) -> Self {
        TtError::Device(e.to_string())
    }
}

impl From<tt_kernels::shard::ShardError> for TtError {
    fn from(e: tt_kernels::shard::ShardError) -> Self {
        TtError::Device(e.to_string())
    }
}

/// The error for failing to open `card`, with what the host can tell about
/// why: no driver, no such card, no permission, or the engine's own text.
pub(crate) fn open_error(card: u16, cause: SessionError) -> TtError {
    use std::io::ErrorKind;
    let kind = match &cause {
        SessionError::Transport(TransportError::Io(io)) => Some(io.kind()),
        _ => None,
    };
    let dir = std::path::Path::new("/dev/tenstorrent");
    match kind {
        Some(ErrorKind::NotFound) if !dir.exists() => TtError::NoDriver,
        Some(ErrorKind::NotFound) => TtError::NoSuchCard {
            card,
            present: tt_kmd::Kmd::enumerate().unwrap_or_default(),
        },
        Some(ErrorKind::PermissionDenied) => TtError::Permission {
            path: dir.join(card.to_string()).display().to_string(),
        },
        _ => cause.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io(kind: std::io::ErrorKind) -> SessionError {
        SessionError::Transport(TransportError::Io(std::io::Error::from(kind)))
    }

    #[test]
    fn a_session_with_no_such_tile_keeps_the_coordinates() {
        let e: TtError = SessionError::NoSuchTile {
            x: 3,
            y: 4,
            columns: vec![1, 2],
        }
        .into();
        assert_eq!(
            e,
            TtError::NoSuchTile {
                x: 3,
                y: 4,
                columns: vec![1, 2]
            }
        );
    }

    #[test]
    fn running_out_of_gddr_is_out_of_memory() {
        let e: TtError = TensorError::OutOfMemory {
            slots: 7,
            free: 3,
            largest: 1,
        }
        .into();
        assert_eq!(
            e,
            TtError::OutOfMemory {
                slots: 7,
                free: 3,
                largest: 1
            }
        );
        let wrapped: TtError = TensorError::Run(RunError::Queued("x".into())).into();
        assert_eq!(wrapped, TtError::Device("x".into()));
    }

    #[test]
    fn a_wait_that_expired_is_a_timeout_and_a_panic_is_not() {
        let expired = WaitError::TimedOut {
            waited_cycles: 10,
            wall_clock: None,
            last_status: 0,
        };
        assert!(matches!(
            TtError::from(RunError::Setup(expired)),
            TtError::Timeout { .. }
        ));
        assert!(matches!(
            TtError::from(RunError::Setup(WaitError::Panicked { code: 3 })),
            TtError::Device(_)
        ));
        assert!(matches!(
            TtError::from(DmError::TimedOut { seq: 1, done: 0 }),
            TtError::Timeout { .. }
        ));
    }

    #[test]
    fn an_op_failure_displays_its_cause_and_reports_it_as_the_root() {
        let cause = Arc::new(TtError::Device("the tile hung".into()));
        let op = TtError::Op {
            what: "matmul".into(),
            device: TtDevice::new(2),
            cause: cause.clone(),
        };
        assert_eq!(op.to_string(), "matmul on tt:2: the tile hung");
        let later = TtError::EarlierFailure(Arc::new(op));
        assert!(later
            .to_string()
            .starts_with("an earlier device op failed: matmul"));
        assert_eq!(later.root(), &*cause);
    }

    #[test]
    fn opening_a_missing_node_is_classified_by_what_the_host_has() {
        // `/dev/tenstorrent` may or may not exist where this runs; either way
        // a `NotFound` for card 0x7FFF is one of the two, never `Device`.
        match open_error(0x7FFF, io(std::io::ErrorKind::NotFound)) {
            TtError::NoDriver => assert!(!std::path::Path::new("/dev/tenstorrent").exists()),
            TtError::NoSuchCard { card, present } => {
                assert_eq!(card, 0x7FFF);
                assert!(!present.contains(&0x7FFF));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            open_error(0, io(std::io::ErrorKind::PermissionDenied)),
            TtError::Permission { .. }
        ));
        assert!(matches!(
            open_error(0, io(std::io::ErrorKind::Other)),
            TtError::Device(_)
        ));
    }
}
