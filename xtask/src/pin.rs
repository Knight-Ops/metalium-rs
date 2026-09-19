//! Pinned upstream revisions, kept in sync with `PINS.toml`.
//!
//! `PINS.toml` is the human-readable record; this is the copy the build reads.
//! Bumping either without the other is caught by the hash checks in [`crate::fetch`].

pub const TTSIM_TAG: &str = "v1.10.9";
pub const TTSIM_ASSET: &str = "libttsim_bh.so";
pub const TTSIM_SHA256: &str = "e6ed2da11718683738d43f14a0bf4f13285b8621697b3165c1eaa240d36cfad5";

/// The tt-metal commit `BackendConfiguration.md:17` cites for `cfg_defines.h`.
pub const TT_METAL_REV: &str = "81989dcdb8f9b340c932ae7a71a346f4f08703eb";
pub const CFG_DEFINES_SHA256: &str =
    "bc2636abc3ea04e6ca322923e6f2713b242857d021d8ef39b9180a14bcb1a5b8";

/// The ISA specification commit. Fetched and verified by `cargo xtask fetch-spec`.
pub const SPEC_REV: &str = "f848eb668c2aeae742a88a49a86157e24a0a20c6";
/// SHA-256 over the specification files the generator reads — not over the
/// tarball, which GitHub does not promise is byte-stable. See `crate::spec`.
pub const SPEC_CONTENT_SHA256: &str =
    "d08a44478ad67fa2c5516366536644741963203925f69b91976fe67eded9ae76";
