//! Packer ReLU and edge masking (lane PU, sub-tranches 1 and 2).
//!
//! A whole FP32 tile goes L1 -> `Dst` (raw bits, `UnpackToDst`) and back out
//! through the packer with one optional stage on, and every datum is compared
//! with an independent raw-bit model of `Packers/ReLU.md` / `EdgeMasking.md`
//! written here, on integers, sharing nothing with `tt_isa::packer`.
//! `Packers/*` is a Wormhole page and `PACR.md` on Blackhole is "basic", so the
//! behaviour is UNVERIFIED on Blackhole: the simulator arms pin what ttsim does
//! (divergence rows 85 and 86), and the `silicon` arms, which this lane has not
//! run, measure the chip against the page.
//!
//! Silicon order (documented/measured first, isolated probes before gates):
//! `silicon_probe_relu_zero_isolated`, `silicon_probe_edge_partial_mask_isolated`,
//! then `packer_relu_matches_the_raw_bit_model`, `relu_unspecified_classes`,
//! `packer_edge_row_masks_match_the_page_lookup`,
//! `silicon_edge_partial_columns_and_negative_infinity`,
//! `silicon_edge_then_relu_order`.
use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::packer::{EdgeFill, EdgeMasking, PackerRelu, PlaneRows, ThresholdFormat};
use tt_isa::tile::{L1Format, TileImage};
use tt_kernels::datapath::{
    config_program, pack_tile_from_dst_staged, state_id, thread_config, tile_descriptor,
    tile_unpack_config, unpack_tile_to_dst, PackStages, OUT, STAGE, TILE_DATUMS,
};
use tt_tests::harness::{self, Roles, Run};

/// Raw FP32 patterns the stages must treat exactly: both zeros, subnormals,
/// infinities, NaN payloads of both signs, extremes, and values around the
/// thresholds the tests use.
const SPECIALS: [u32; 34] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // smallest positive subnormal
    0x8000_0001,
    0x007f_ffff, // largest subnormal
    0x807f_ffff,
    0x0080_0000, // smallest normal
    0x8080_0000,
    0x3f80_0000, // 1.0
    0xbf80_0000,
    0x3f7f_ffff, // just below 1.0
    0x3f80_0001, // just above 1.0
    0xbf7f_ffff,
    0xbf80_0001,
    0x4000_0000, // 2.0
    0xc000_0000,
    0x4040_0000, // 3.0
    0x3f00_0000, // 0.5
    0xbf00_0000,
    0x7f7f_ffff, // max
    0xff7f_ffff,
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // quiet NaN
    0xffc0_0000,
    0x7fc1_2345, // NaN payloads
    0xffc1_2345,
    0x7f80_0001, // signalling NaN
    0xff80_0001,
    0x3f80_8000, // 1.0 + half a BF16 ulp: low 16 bits non-zero
    0xbf80_8000,
    0x3f81_0000, // next BF16 above 1.0
    0x4020_0000, // 2.5
    0x0001_0000, // subnormal with a bit above the BF16 cut
];

fn tile_bits() -> Vec<u32> {
    let mut s = 0x1234_5679u32;
    (0..TILE_DATUMS as usize)
        .map(|i| {
            if i % 3 == 0 {
                return SPECIALS[(i / 3) % SPECIALS.len()];
            }
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Normal of either sign over a wide exponent range.
            (s & 0x8000_0000) | (((s >> 8) % 250 + 2) << 23) | (s & 0x7f_ffff)
        })
        .collect()
}

fn image(bits: &[u32]) -> Vec<u8> {
    let img = TileImage::new(tile_descriptor(), L1Format::Fp32).unwrap();
    let mut b = vec![0u8; img.total_bytes()];
    for (i, d) in bits.iter().enumerate() {
        let at = img.datum_bit_offset(i) / 8;
        b[at..at + 4].copy_from_slice(&d.to_le_bytes());
    }
    b
}

/// Bytes of sentinel around the packed tiles: the packer writes exactly the
/// tiles it is given, so the guards must come back untouched.
const GUARD: usize = 64;

/// Everything off, every word written where the chip (unlike ttsim) can take it.
fn restored() -> PackStages {
    PackStages {
        exhaustive: cfg!(feature = "silicon"),
        ..PackStages::OFF
    }
}

/// The program of an L1 -> `Dst` -> packer run: `stages` for the first tile, the
/// restored (off) stages for a second, `Dst` read from row 0 both times.
fn pack_roles(
    stages: &PackStages,
) -> (Vec<tt_isa::isa::Instruction>, Vec<tt_isa::isa::Instruction>) {
    let mut unpack = thread_config();
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, STAGE);
    unpack.extend(config_program(&words));
    unpack.extend(unpack_tile_to_dst(STAGE, 0));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());

    let mut pack = vec![state_id()];
    pack.extend(pack_tile_from_dst_staged(OUT, 0, stages).unwrap());
    // Put the stages back to `Config`'s reset state: silicon keeps `Config`
    // across programs.
    pack.extend(pack_tile_from_dst_staged(OUT + 4096, 0, &restored()).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    (unpack, pack)
}

/// Run [`pack_roles`] and hand `check` the first tile (1024 raw words), inside
/// the forked child where the simulator lives. The second tile must be the
/// input again and both guard regions (`GUARD` bytes of `0xA5` before and
/// after) untouched; those are asserted here.
fn run_pack(bits: &[u32], stages: &PackStages, check: impl FnOnce(&[u32])) {
    let (unpack, pack) = pack_roles(stages);
    let input = image(bits);
    let sentinel = vec![0xA5u8; 2 * 4096 + 2 * GUARD];
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &[],
                pack: &pack,
            })
            .stage(&[(STAGE, &input), (OUT - GUARD as u64, &sentinel)])
            .read_back(&[(OUT - GUARD as u64, sentinel.len())]),
        );
        let all = &out.l1[0];
        assert!(
            all[..GUARD].iter().all(|&b| b == 0xA5),
            "guard before OUT written"
        );
        assert!(
            all[GUARD + 8192..].iter().all(|&b| b == 0xA5),
            "guard after the second tile written"
        );
        let words: Vec<u32> = all[GUARD..GUARD + 8192]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            &words[1024..],
            bits,
            "stages off after restore: pass-through"
        );
        check(&words[..1024]);
    });
}

/// Whether the same program survives the simulator (a refusal `_Exit`s the child).
fn pack_survives(bits: &[u32], stages: &PackStages) -> bool {
    let (unpack, pack) = pack_roles(stages);
    let input = image(bits);
    harness::survives(|dev| {
        harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &[],
                pack: &pack,
            })
            .stage(&[(STAGE, &input)]),
        );
    })
}

// ---- independent raw-bit oracle: ReLU --------------------------------------

fn is_nan(x: u32) -> bool {
    x & 0x7fff_ffff > 0x7f80_0000
}

/// Total order on non-NaN floats as sign-magnitude integers: -0 == +0.
fn key(x: u32) -> i64 {
    let mag = i64::from(x & 0x7fff_ffff);
    if x >> 31 == 1 {
        -mag
    } else {
        mag
    }
}

/// `x <= t` with the IEEE meaning, NaN false.
fn le(x: u32, t: u32) -> bool {
    !is_nan(x) && !is_nan(t) && key(x) <= key(t)
}

#[derive(Copy, Clone, Debug)]
enum Model {
    Zero,
    Min(u16),
    Max(u16),
}

/// `Packers/ReLU.md`'s `ReLUStage`, on FP32 raw bits with a BF16 threshold.
fn relu_model(m: Model, x: u32) -> u32 {
    match m {
        Model::Zero => {
            if le(x, 0) {
                0
            } else {
                x
            }
        }
        Model::Min(t) => {
            if le(x, u32::from(t) << 16) {
                0
            } else {
                x
            }
        }
        Model::Max(t) => {
            let thr = u32::from(t) << 16;
            if le(x, 0) {
                0
            } else if !is_nan(x) && key(x) > key(thr) {
                thr
            } else {
                x
            }
        }
    }
}

/// Datums whose result the page defines *and* ttsim reproduces: not NaN (ttsim
/// tests the sign bit and compares as integers: row 85) and, for the clamp, not
/// within a BF16 ulp above the threshold, where ttsim leaves `0x3f800001`
/// unclamped under `Max(0x3f80)` and clamps `0x3f807fff`.
fn comparable(m: Model, x: u32) -> bool {
    if is_nan(x) {
        return false;
    }
    if let Model::Max(t) = m {
        let thr = u32::from(t) << 16;
        if x > thr && x < thr + 0x8000 {
            return false;
        }
    }
    true
}

fn stages_for(m: Model) -> PackStages {
    let relu = match m {
        Model::Zero => PackerRelu::ZERO,
        Model::Min(t) => PackerRelu::min_threshold(t, ThresholdFormat::Bf16).unwrap(),
        Model::Max(t) => PackerRelu::max_threshold(t, ThresholdFormat::Bf16).unwrap(),
    };
    PackStages { relu, ..restored() }
}

const RELU_MODES: [Model; 7] = [
    Model::Zero,
    Model::Min(0x0000),
    Model::Min(0x3f80),
    Model::Min(0x7f80), // +inf: everything non-NaN is replaced by zero
    Model::Max(0x3f80),
    Model::Max(0x4020),
    Model::Max(0x7f80),
];

#[test]
fn packer_relu_matches_the_raw_bit_model() {
    let bits = tile_bits();
    for m in RELU_MODES {
        run_pack(&bits, &stages_for(m), |got| {
            let mut checked = 0;
            for (i, (&x, &g)) in bits.iter().zip(got).enumerate() {
                if !comparable(m, x) {
                    continue;
                }
                checked += 1;
                let w = relu_model(m, x);
                assert_eq!(
                    g, w,
                    "{m:?}: datum {i}: in {x:#010x}, got {g:#010x}, model {w:#010x}"
                );
            }
            assert!(checked > 700, "{m:?}: only {checked} datums compared");
        });
    }
}

/// The oracle must distinguish the modes from each other and from a pass
/// through, on this data, or a gate that matched it could match a broken stage.
#[test]
fn relu_oracle_distinguishes_the_modes() {
    let bits = tile_bits();
    let differs = |a: Model, b: Model| {
        bits.iter()
            .filter(|&&x| comparable(a, x) && comparable(b, x))
            .any(|&x| relu_model(a, x) != relu_model(b, x))
    };
    assert!(bits.iter().any(|&x| relu_model(Model::Zero, x) != x));
    assert!(differs(Model::Zero, Model::Min(0x3f80)));
    assert!(differs(Model::Min(0x3f80), Model::Max(0x3f80)));
    assert!(differs(Model::Max(0x3f80), Model::Max(0x4020)));
    // -0 and negative values become +0, not -0; NaNs pass with their payload.
    assert_eq!(relu_model(Model::Zero, 0x8000_0000), 0);
    assert_eq!(relu_model(Model::Zero, 0xffc1_2345), 0xffc1_2345);
    assert_eq!(relu_model(Model::Max(0x3f80), 0x7f80_0000), 0x3f80_0000);
    // A datum equal to the threshold is replaced under Min (`<=`, not `<`).
    assert_eq!(relu_model(Model::Min(0x3f80), 0x3f80_0000), 0);
    // Each exclusion in `comparable` removes something the page defines.
    assert!(is_nan(0x7fc1_2345) && relu_model(Model::Max(0x3f80), 0x7fc1_2345) == 0x7fc1_2345);
}

/// What the page leaves to the float semantics of NaN, and the Max clamp's
/// sub-BF16 neighbourhood: pinned on ttsim (divergence row 85), measured on
/// silicon.
#[test]
fn relu_unspecified_classes() {
    const PROBES: [u32; 16] = [
        0x3f7f_ffff,
        0x3f80_0000,
        0x3f80_0001,
        0x3f80_7fff,
        0x3f80_8000,
        0x3f81_0000,
        0x7f80_0000,
        0x7fc0_0000,
        0x7fc1_2345,
        0x7f80_0001,
        0xff80_0001,
        0xffc0_0000,
        0xffc1_2345,
        0x8000_0001,
        0x0000_0001,
        0xffff_ffff,
    ];
    let bits: Vec<u32> = (0..1024).map(|i| PROBES[i % 16]).collect();
    for m in [Model::Zero, Model::Min(0x3f80), Model::Max(0x3f80)] {
        run_pack(&bits, &stages_for(m), |got| {
            for (k, &x) in PROBES.iter().enumerate() {
                let g = got[k];
                if cfg!(feature = "silicon") {
                    println!(
                        "MEASURED relu {m:?} {x:#010x} -> {g:#010x} (page {:#010x})",
                        relu_model(m, x)
                    );
                    if is_nan(x) || !comparable(m, x) {
                        let thr = match m {
                            Model::Max(t) | Model::Min(t) => u32::from(t) << 16,
                            Model::Zero => 0,
                        };
                        assert!(
                            g == x || g == 0 || g == thr || g == relu_model(m, x),
                            "{m:?} {x:#010x}: {g:#010x} is neither passed through, zero, the \
                             threshold nor the page's value"
                        );
                    }
                    continue;
                }
                // ttsim: a sign-bit test, then integer compares; the clamp
                // reaches only datums a BF16 ulp or so above the threshold.
                let observed = match (m, x) {
                    (_, x) if x >> 31 == 1 => 0,
                    (Model::Zero, x) => x,
                    (Model::Min(t), x) => {
                        if x <= u32::from(t) << 16 {
                            0
                        } else {
                            x
                        }
                    }
                    (Model::Max(_), 0x3f80_0001) => 0x3f80_0001,
                    (Model::Max(t), x) => {
                        if x > u32::from(t) << 16 {
                            u32::from(t) << 16
                        } else {
                            x
                        }
                    }
                };
                assert_eq!(g, observed, "ttsim {m:?} {x:#010x}");
            }
        });
    }
}

// ---- independent raw-bit oracle: edge masking ------------------------------

/// The edge-masking test vector, as plain integers: no `tt_isa::packer` type
/// in the oracle's inputs.
#[derive(Copy, Clone)]
struct Edge {
    fill: EdgeFill,
    masks: [u16; 4],
    row_set: usize,
    mapping: [[usize; 16]; 4],
    plane: usize,
}

/// `Packers/EdgeMasking.md` on the row-set path, on Blackhole's single packer
/// (`PACR.md`: packer 0 only, the `k`'th enabled read interface at tile row
/// `TilePosition + k`, wrapping at `pack_reads_per_xy_plane`): the datum at tile
/// row `row`, column `col` is kept iff its mask has bit `col` set.
fn edge_keeps(e: &Edge, row: usize, col: usize) -> bool {
    let y = row % e.plane;
    let c = e.mapping[e.row_set][y];
    e.masks[c] >> col & 1 == 1
}

fn edge_stages(e: &Edge) -> PackStages {
    PackStages {
        edge: EdgeMasking {
            fill: e.fill,
            column_masks: e.masks,
            row_set: e.row_set as u8,
            row_set_mapping: e.mapping.map(|m| m.map(|c| c as u8)),
            plane_rows: if e.plane == 16 {
                PlaneRows::Sixteen
            } else {
                PlaneRows::Four
            },
        },
        ..restored()
    }
}

/// Input datums that are never zero or -inf and all distinct, so a kept datum,
/// a zeroed one and a -inf one cannot be confused with each other or a neighbour.
fn edge_input() -> Vec<u32> {
    (0..1024u32).map(|i| 0x3f80_0000 + i * 7 + 1).collect()
}

fn check_edge(e: Edge) {
    let bits = edge_input();
    run_pack(&bits, &edge_stages(&e), |got| {
        let mut kept = 0;
        let mut masked = 0;
        for row in 0..64 {
            for col in 0..16 {
                let i = row * 16 + col;
                let want = if edge_keeps(&e, row, col) {
                    kept += 1;
                    bits[i]
                } else {
                    masked += 1;
                    match e.fill {
                        EdgeFill::Zero => 0,
                        EdgeFill::NegativeInfinity => 0xff80_0000,
                    }
                };
                assert_eq!(
                    got[i], want,
                    "row {row} col {col}: got {:#010x}, model {want:#010x}",
                    got[i]
                );
            }
        }
        // Pass-through and all-masked are legitimate patterns; the mixed ones are
        // shown to mix by `edge_oracle_pattern_is_sensitive_to_the_lookup`.
        let _ = (kept, masked);
    });
}

/// Row masks ttsim's pack path accepts: mask sets 0 (all columns) and 1 (none),
/// the only two it models words for with its two accepted values, with row sets
/// 0 and 1 mapping `Y` differently -- so a wrong row-set select, a wrong `Y` or
/// a wrong plane size each lands on a different row.
fn ttsim_pattern(plane: usize, row_set: usize) -> Edge {
    let mut mapping = [[0; 16]; 4];
    mapping[0] = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0];
    mapping[1] = [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0];
    Edge {
        fill: EdgeFill::Zero,
        masks: [0xffff, 0, 0, 0],
        row_set,
        mapping,
        plane,
    }
}

#[test]
fn packer_edge_row_masks_match_the_page_lookup() {
    for plane in [4, 16] {
        for row_set in [0, 1] {
            check_edge(ttsim_pattern(plane, row_set));
        }
    }
    // All-kept and all-masked bracket it: the stage is really configured.
    let mut all_kept = ttsim_pattern(4, 0);
    all_kept.mapping = [[0; 16]; 4];
    check_edge(all_kept);
    all_kept.masks = [0; 4];
    check_edge(all_kept);
}

/// The pattern must not be one a wrong lookup also produces: the other row set,
/// the other plane size, or an ignored `Y` changes it.
#[test]
fn edge_oracle_pattern_is_sensitive_to_the_lookup() {
    let pattern = |e: &Edge| (0..64).map(|r| edge_keeps(e, r, 0)).collect::<Vec<_>>();
    let right = pattern(&ttsim_pattern(16, 1));
    assert_ne!(right, pattern(&ttsim_pattern(16, 0)), "row set");
    assert_ne!(right, pattern(&ttsim_pattern(4, 1)), "plane size");
    let mut no_y = ttsim_pattern(16, 1);
    no_y.mapping[1] = [no_y.mapping[1][0]; 16];
    assert_ne!(right, pattern(&no_y), "Y ignored");
    // Both kept and masked rows exist, and planes repeat (Z does not enter).
    assert!(right.iter().any(|&k| k) && right.iter().any(|&k| !k));
    assert_eq!(right[..16], right[16..32]);
}

/// ttsim refuses what it has not tested, by name, and takes the control.
/// Divergence row 86.
#[cfg(not(feature = "silicon"))]
#[test]
fn ttsim_refuses_partial_masks_and_negative_infinity() {
    let bits = edge_input();
    let control = |mask: u16, fill| {
        let mut e = ttsim_pattern(4, 0);
        e.mapping = [[0; 16]; 4];
        e.masks = [mask, 0, 0, 0];
        e.fill = fill;
        pack_survives(&bits, &edge_stages(&e))
    };
    assert!(control(0xffff, EdgeFill::Zero), "full mask is the control");
    assert!(control(0, EdgeFill::Zero), "empty mask is accepted");
    assert!(
        !control(0x00ff, EdgeFill::Zero),
        "tensix_pacr: edge_mask=0xff"
    );
    assert!(!control(0x7fff, EdgeFill::Zero), "edge_mask=0x7fff");
    assert!(!control(0xfffe, EdgeFill::Zero), "edge_mask=0xfffe");
    assert!(
        !control(0xffff, EdgeFill::NegativeInfinity),
        "tensix_pacr: edge_mode=1"
    );
}

// ---- silicon-only arms (written, not run by the lane) ----------------------

/// Isolated first contact: ReLU zero mode on one tile, nothing asserted but that
/// the program completes. Risk: UNVERIFIED (Wormhole page; Blackhole's
/// `STACC_RELU_ApplyRelu` is a four-bit field).
#[cfg(feature = "silicon")]
#[test]
fn silicon_probe_relu_zero_isolated() {
    let bits = tile_bits();
    assert!(pack_survives(&bits, &stages_for(Model::Zero)));
}

/// Isolated first contact: one partial column mask. Risk: UNVERIFIED.
#[cfg(feature = "silicon")]
#[test]
fn silicon_probe_edge_partial_mask_isolated() {
    let bits = edge_input();
    let mut e = ttsim_pattern(4, 0);
    e.mapping = [[0; 16]; 4];
    e.masks = [0x00ff, 0, 0, 0];
    assert!(pack_survives(&bits, &edge_stages(&e)));
}

#[cfg(feature = "silicon")]
fn four_set_pattern(plane: usize, row_set: usize, fill: EdgeFill) -> Edge {
    let mut mapping = [[0; 16]; 4];
    mapping[0] = [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0];
    mapping[1] = [1, 1, 3, 0, 2, 2, 3, 0, 1, 1, 3, 0, 2, 2, 3, 0];
    mapping[2] = [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
    mapping[3] = [3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2];
    Edge {
        fill,
        masks: [0xffff, 0x00ff, 0xaaaa, 0x0000],
        row_set,
        mapping,
        plane,
    }
}

/// Four mask sets, partial columns, all four row sets, both plane sizes, both
/// fills; ttsim refuses all of it, so this is the page's model against the chip.
#[cfg(feature = "silicon")]
#[test]
fn silicon_edge_partial_columns_and_negative_infinity() {
    for fill in [EdgeFill::Zero, EdgeFill::NegativeInfinity] {
        for plane in [4, 16] {
            for row_set in 0..4 {
                check_edge(four_set_pattern(plane, row_set, fill));
            }
        }
    }
}

/// Edge masking happens before ReLU (`ExponentHistogram.md` lists format
/// conversion, edge masking, ReLU in that order): a -inf fill followed by ReLU
/// zero comes out +0, and a ReLU that ran first would leave -inf.
#[cfg(feature = "silicon")]
#[test]
fn silicon_edge_then_relu_order() {
    let bits = edge_input();
    let e = four_set_pattern(4, 0, EdgeFill::NegativeInfinity);
    let mut stages = edge_stages(&e);
    stages.relu = PackerRelu::ZERO;
    run_pack(&bits, &stages, |got| {
        for row in 0..64 {
            for col in 0..16 {
                let want = if edge_keeps(&e, row, col) {
                    bits[row * 16 + col]
                } else {
                    0
                };
                assert_eq!(got[row * 16 + col], want, "row {row} col {col}");
            }
        }
    });
}
