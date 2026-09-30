//! Pinned upstream revisions, kept in sync with `PINS.toml`.
//!
//! `PINS.toml` is the human-readable record; this is the copy the build reads.
//! Bumping either without the other is caught by the hash checks in [`crate::fetch`].

pub const TTSIM_TAG: &str = "v1.10.9";

/// One release asset of the pinned ttsim tag.
pub struct TtsimAsset {
    pub name: &'static str,
    pub sha256: &'static str,
}

/// The simulator builds this repository fetches, all from [`TTSIM_TAG`].
///
/// One tag covers every build, so the single-chip and dual-chip simulators
/// cannot drift apart. `bh_x4` joined for Phase 8's sharding gate; `bh_x32`
/// exists in the same release and is deliberately not pinned until something
/// needs it.
pub const TTSIM_ASSETS: &[TtsimAsset] = &[
    TtsimAsset {
        name: "libttsim_bh.so",
        sha256: "e6ed2da11718683738d43f14a0bf4f13285b8621697b3165c1eaa240d36cfad5",
    },
    TtsimAsset {
        name: "libttsim_bh_x2.so",
        sha256: "e1ffbaf39c7d071a2f64d7072a586da5f9a7bef55907459edbdb185189d23108",
    },
    TtsimAsset {
        name: "libttsim_bh_x4.so",
        sha256: "942a7a2a210259b77d1c44a7b13ac98fe672aec28d28e4385fcd30f6967756c1",
    },
];

/// The tt-metal commit `BackendConfiguration.md:17` cites for `cfg_defines.h`.
pub const TT_METAL_REV: &str = "81989dcdb8f9b340c932ae7a71a346f4f08703eb";
pub const CFG_DEFINES_SHA256: &str =
    "bc2636abc3ea04e6ca322923e6f2713b242857d021d8ef39b9180a14bcb1a5b8";

/// The ISA specification commit. Fetched and verified by `cargo xtask fetch-spec`.
pub const SPEC_REV: &str = "f848eb668c2aeae742a88a49a86157e24a0a20c6";
/// SHA-256 over the specification files the generator reads — not over the
/// tarball, which GitHub does not promise is byte-stable. See `crate::spec`.
pub const SPEC_CONTENT_SHA256: &str =
    "218cc3281f0216070c2586db23cc747b3ba3ae0b23de81114cbd15820abbfc30";

/// The tt-kmd release whose `ioctl.h` defines the userspace ABI `tt-kmd` binds.
///
/// Chosen to match the driver actually loaded rather than to be current: at this
/// tag the header is byte-identical to `/usr/src/tenstorrent-2.11.0/ioctl.h`, the
/// DKMS source for the running module. A pin that named a newer tag would compile
/// and would be wrong.
pub const TTKMD_TAG: &str = "ttkmd-2.11.0";
pub const TTKMD_IOCTL_SHA256: &str =
    "831031cdb556f3f13597af838535d4ce594b47b63d685e87312db76087fbd4dc";

// `TENSTORRENT_DRIVER_VERSION` deliberately does not appear here. The hash above
// pins what we compiled against; what we are *talking to* is pinned by
// `tt_kmd::PINNED_API_VERSION`, which `Kmd::open` checks against the driver's own
// `GET_DRIVER_INFO` answer, because a matching header is no evidence about the
// module that is loaded. A second copy of the number in this file would be a
// second source of truth for one fact, and the copy nothing asserts is the one
// that silently goes stale.

/// MNIST, for the Phase 7 training gate. Fetched by `cargo xtask fetch-mnist`
/// from the Google CVDF mirror (the original host is unreliable), verified
/// against both the published `.gz` hash and the decompressed IDX file's.
pub struct MnistFile {
    pub name: &'static str,
    pub gz_sha256: &'static str,
    pub sha256: &'static str,
}

pub const MNIST_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";

pub const MNIST_FILES: &[MnistFile] = &[
    MnistFile {
        name: "train-images-idx3-ubyte",
        gz_sha256: "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        sha256: "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
    },
    MnistFile {
        name: "train-labels-idx1-ubyte",
        gz_sha256: "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        sha256: "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
    },
    MnistFile {
        name: "t10k-images-idx3-ubyte",
        gz_sha256: "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        sha256: "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    },
    MnistFile {
        name: "t10k-labels-idx1-ubyte",
        gz_sha256: "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        sha256: "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
    },
];
