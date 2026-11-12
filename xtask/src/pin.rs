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

/// The Burn tag whose `crates/burn-backend-tests/tests/` is the conformance suite
/// `burn-tt-conformance` runs. Fetched and verified by `cargo xtask fetch-burn-tests`.
/// The same release as the `=0.21.0` crates the workspace compiles against.
pub const BURN_TESTS_TAG: &str = "v0.21.0";
/// SHA-256 over every file of the extracted `tests/` tree, not over the tarball,
/// which GitHub does not promise is byte-stable. See `crate::burn_tests`.
pub const BURN_TESTS_CONTENT_SHA256: &str =
    "45c5b2f9c8903944bbfe0af1e31ef929377e2ec5e705000204a57acacfff9d16";

/// One pretrained-weights file of the model ladder (`PINS.toml`, `[weights]`).
/// Fetched by `cargo xtask fetch-weights` into `vendor/weights/<model>/<file>`.
pub struct WeightFile {
    /// The short name `tt-models` and `--model` use.
    pub model: &'static str,
    /// Hugging Face repository.
    pub repo: &'static str,
    /// A commit of that repository, never a branch.
    pub revision: &'static str,
    pub file: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    /// Fetched only by `--model <model>`, never by a bare `fetch-weights`: the multi-gigabyte
    /// checkpoints of the architectures past Llama, which no routine gate reads.
    pub on_request: bool,
}

pub const WEIGHTS_URL: &str = "https://huggingface.co";

pub const WEIGHT_FILES: &[WeightFile] = &[
    WeightFile {
        model: "resnet18",
        repo: "timm/resnet18.a1_in1k",
        revision: "491b427b45c94c7fb0e78b5474cc919aff584bbf",
        file: "model.safetensors",
        bytes: 46_807_446,
        sha256: "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612",
        on_request: false,
    },
    WeightFile {
        model: "vit-tiny",
        repo: "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        revision: "7d3afdd0cf93ad84d986eb2d6bcc5812ebd0b106",
        file: "model.safetensors",
        bytes: 22_883_348,
        sha256: "fecf81b492bd13ee7a5297cb74d1d417aac8bf7e1b7d96aed89c4691984587ed",
        on_request: false,
    },
    WeightFile {
        model: "smollm2-135m",
        repo: "HuggingFaceTB/SmolLM2-135M",
        revision: "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        file: "model.safetensors",
        bytes: 269_060_552,
        sha256: "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
        on_request: false,
    },
    WeightFile {
        model: "smollm2-1.7b-instruct",
        repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision: "31b70e2e869a7173562077fd711b654946d38674",
        file: "model.safetensors",
        bytes: 3_422_777_952,
        sha256: "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
        on_request: false,
    },
    // Qwen3.5-4B: hybrid Gated DeltaNet and gated attention (`tt_models::qwen35`), BF16 on disk.
    WeightFile {
        model: "qwen3.5-4b",
        repo: "Qwen/Qwen3.5-4B",
        revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        file: "model.safetensors-00001-of-00002.safetensors",
        bytes: 5_329_398_688,
        sha256: "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
        on_request: true,
    },
    WeightFile {
        model: "qwen3.5-4b",
        repo: "Qwen/Qwen3.5-4B",
        revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        file: "model.safetensors-00002-of-00002.safetensors",
        bytes: 3_990_429_408,
        sha256: "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
        on_request: true,
    },
    // NVIDIA-Nemotron-3.5-Bolt-E4B: Gemma 4 E4B's text model (`tt_models::gemma4`), BF16 on disk.
    // The checkpoint's second file, `model-frozen-shared-kv.safetensors`, holds only the key/value
    // projections of the layers that share another layer's keys and values, which nothing reads.
    WeightFile {
        model: "nemotron-3.5-bolt-e4b",
        repo: "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16",
        revision: "49843eed8db1f1c616c47af75cfd36e07aec0667",
        file: "model-00001-of-00001.safetensors",
        bytes: 15_882_494_698,
        sha256: "15006dbbfc99dbde9cc3fb5742b08976837b6ddc9db05b1ceb8b223de8cfc520",
        on_request: true,
    },
    // MOSS-Image-Base: a Qwen3.5 text encoder, a diffusion transformer and the FLUX.2 VAE
    // (`tt_models::moss_image`), each in its subdirectory of the repository.
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "text_encoder/model-00001-of-00004.safetensors",
        bytes: 4_973_910_744,
        sha256: "abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b",
        on_request: true,
    },
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "text_encoder/model-00002-of-00004.safetensors",
        bytes: 4_988_364_288,
        sha256: "9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894",
        on_request: true,
    },
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "text_encoder/model-00003-of-00004.safetensors",
        bytes: 4_954_202_112,
        sha256: "956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc",
        on_request: true,
    },
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "text_encoder/model-00004-of-00004.safetensors",
        bytes: 1_869_001_848,
        sha256: "9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208",
        on_request: true,
    },
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        bytes: 4_991_978_016,
        sha256: "ae085fba123b6df67daf90aa9698fadee60c73e45bcaf3f3043d15f34e7a1804",
        on_request: true,
    },
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        bytes: 4_983_137_016,
        sha256: "806e876d80c7fba560663a3a95580e6c7033da3f9ea3c52765245ef0e1818f75",
        on_request: true,
    },
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        bytes: 189_549_184,
        sha256: "5f319b72c40324294007c270d78112425627b537fcd8769b3c00f9f070e506a9",
        on_request: true,
    },
    WeightFile {
        model: "moss-image-base",
        repo: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        file: "vae/diffusion_pytorch_model.safetensors",
        bytes: 249_521_372,
        sha256: "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a",
        on_request: true,
    },
];

/// One pinned test image of the model ladder (`PINS.toml`, `[[weights.images]]`), fetched by
/// `cargo xtask fetch-weights --model images` into `vendor/weights/images/<name>`.
pub struct ImageFile {
    pub name: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

pub const IMAGE_FILES: &[ImageFile] = &[
    ImageFile {
        name: "cats.jpg",
        url: "http://images.cocodataset.org/val2017/000000039769.jpg",
        bytes: 173_131,
        sha256: "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e",
    },
    ImageFile {
        name: "dog.jpg",
        url: "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        bytes: 661_378,
        sha256: "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a",
    },
    ImageFile {
        name: "bear.jpg",
        url: "http://images.cocodataset.org/val2017/000000000285.jpg",
        bytes: 335_861,
        sha256: "f3a2974ce3686332609124c70e3e6a2e3aca43fccf1cd1bd7c5c03820977f57d",
    },
    ImageFile {
        name: "living_room.jpg",
        url: "http://images.cocodataset.org/val2017/000000000139.jpg",
        bytes: 161_811,
        sha256: "ffe0f0cec3b2e27aab1967229cdf0a0d7751dcdd5800322f0b8ac0dffb3b8a8d",
    },
    ImageFile {
        name: "bedroom.jpg",
        url: "http://images.cocodataset.org/val2017/000000000632.jpg",
        bytes: 155_667,
        sha256: "a4cd7f45ac1ce27eaafb254b23af7c0b18a064be08870ceaaf03b2147f2ce550",
    },
];

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
