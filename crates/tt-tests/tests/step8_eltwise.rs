//! Phase 5: a real kernel — unpack, compute on the Vector Unit, pack.
//!
//! The datapath is the one `probe_unpack.rs` and `probe_pack.rs` established; what
//! is new is SFPU instructions in the middle of it, and an expected value that
//! comes from the specification's own arithmetic model rather than from the host's.
//!
//! # The tolerance policy, concretely
//!
//! There is no tolerance. `tt_isa::numerics::fma_bh` is a port of
//! `Miscellaneous/FMA/fma.c`'s `fma_model_bh`, which
//! `Miscellaneous/FMA/README.md` says matches "Blackhole Tensix Vector Unit (SFPU)
//! `SFPMAD` family of instructions". `tests/fma_oracle.rs` checks the port against
//! the C itself over 200 000 cases. So every expected value here is a bit pattern,
//! asserted with `assert_eq`, and a mismatch is a finding rather than noise.
//!
//! # Why FP32 only
//!
//! ttsim declines `UnpackToDst` for every 16-bit and block-float input format
//! (`docs/ttsim-divergence.md` row 31), so BF16 cannot reach `Dst` through the
//! unpacker on the simulator at all. The BF16 kernel is silicon-only until the
//! packer offers another route into `Dst`.

use tt_isa::backend;
use tt_isa::isa::Instruction;
use tt_isa::numerics;
use tt_isa::sfpu::{self, mod0_fmt, DST_ODD_COLUMNS};
use tt_isa::tile::TileDescriptor;
use tt_tests::datapath::{flat_descriptor, staged_image};
use tt_tests::harness::in_device;

mod eltwise_support;
use eltwise_support::{datum_at_dst_flat, run_kernel, DATUMS, GROUP_DATUMS};

/// `LReg` indices this file uses. `LReg[8]` upwards are constants, so the working
/// set is 0..=7 (`tt_isa::sfpu::MAX_WRITABLE_LREG`).
const A: u32 = 0;
const B: u32 = 1;
const RESULT: u32 = 2;

/// The `Dst` address of the second operand's row group, in `SFPLOAD` address units.
///
/// `SFPLOAD`'s address selects an aligned group of four rows via its top bits, so
/// stepping by 4 moves one group -- 64 datums -- along.
const B_GROUP: u32 = 4;

/// Which flat `Dst` position lane `lane` of an `SFPLOAD`/`SFPSTORE` touches.
///
/// `SFPLOAD.md`: `Row = (Addr & ~3) + Lane/8`, `Column = (Lane & 7) * 2`, plus one
/// if the address has [`DST_ODD_COLUMNS`] set. One instruction therefore reaches 32
/// of the 64 datums in a four-row group, and the two parities together cover it.
fn lane_flat(addr: u32, lane: usize) -> usize {
    let row = (addr as usize & !3) + lane / 8;
    let column = (lane & 7) * 2 + usize::from(addr & DST_ODD_COLUMNS != 0);
    row * 16 + column
}

/// One SFPU pass over one column parity of the first four `Dst` rows.
///
/// Operand A is read from row group 0 and operand B from row group [`B_GROUP`];
/// the result overwrites A, which is what the packer then reads.
fn pass(
    columns: u32,
    op: fn(u32, u32, u32) -> Result<Instruction, sfpu::EncodeError>,
) -> Vec<Instruction> {
    vec![
        sfpu::load(A, mod0_fmt::FP32, 0, columns).unwrap(),
        sfpu::load(B, mod0_fmt::FP32, 0, B_GROUP | columns).unwrap(),
        op(A, B, RESULT).unwrap(),
        sfpu::store(RESULT, mod0_fmt::FP32, 0, columns).unwrap(),
    ]
}

/// The kernel: both column parities of the first four `Dst` rows.
fn binary_kernel(
    op: fn(u32, u32, u32) -> Result<Instruction, sfpu::EncodeError>,
) -> Vec<Instruction> {
    let mut p = pass(0, op);
    p.extend(pass(DST_ODD_COLUMNS, op));
    p
}

/// `staged[i] = values[i]`, as an L1 tile image.
fn staged_values(descriptor: TileDescriptor, values: &[f32]) -> Vec<u8> {
    let mut bytes = staged_image(descriptor, 0);
    let image = tt_isa::tile::TileImage::new(descriptor, tt_isa::tile::L1Format::Fp32).unwrap();
    for (i, v) in values.iter().enumerate() {
        let off = image.datum_bit_offset(i) / 8;
        bytes[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    bytes
}

/// Check every packed word against `model`, applied to the two staged datums the
/// lane mapping says that word's lanes read.
///
/// Returns how many words were checked, so a gate can assert the comparison was not
/// vacuous.
#[track_caller]
fn check_against_model(packed: &[u32], values: &[f32], model: fn(u32, u32) -> u32) -> usize {
    let mut checked = 0;
    for (flat, &_got) in packed.iter().enumerate().take(GROUP_DATUMS) {
        // The first four positions hold no staged datum, so operand A there is
        // `Dst`'s power-on value -- `UnpredictableValue` per `Dst.md:15`. Excluded
        // rather than asserted against the zero ttsim happens to give.
        let (Some(a_idx), Some(b_idx)) = (
            datum_at_dst_flat(flat),
            datum_at_dst_flat(flat + GROUP_DATUMS),
        ) else {
            continue;
        };
        if a_idx >= values.len() || b_idx >= values.len() {
            continue;
        }
        let want = model(values[a_idx].to_bits(), values[b_idx].to_bits());
        assert_eq!(
            packed[flat],
            want,
            "Dst[{}][{}]: {:#010x} op {:#010x} gave {:#010x}, the model says {want:#010x}",
            flat / 16,
            flat % 16,
            values[a_idx].to_bits(),
            values[b_idx].to_bits(),
            packed[flat]
        );
        checked += 1;
    }
    checked
}

/// 128 distinct values: the first 64 are operand A, the next 64 operand B.
fn operands() -> Vec<f32> {
    (0..2 * GROUP_DATUMS)
        .map(|i| {
            // Distinct, of mixed sign, and with low mantissa bits set, so a result
            // that matched by coincidence would be remarkable.
            let x = i as f32;
            (x - 63.5) * 0.375 + if i % 3 == 0 { 0.125 } else { -0.0625 }
        })
        .collect()
}

/// The kernel's two passes together touch every datum of the group, exactly once.
///
/// `SFPLOAD` and `SFPSTORE` each reach 32 of a four-row group's 64 datums, so a
/// kernel that ran only one pass would silently leave half the tile unwritten --
/// and the numeric gates would not notice, because they only check positions the
/// staged operands reached. This pins the coverage claim separately, from the
/// addressing model rather than from a run.
#[test]
fn the_two_passes_cover_every_datum_of_the_group_exactly_once() {
    let mut seen = vec![0u32; GROUP_DATUMS];
    for addr in [0, DST_ODD_COLUMNS] {
        for lane in 0..32 {
            seen[lane_flat(addr, lane)] += 1;
        }
    }
    assert!(
        seen.iter().all(|&n| n == 1),
        "each of the {GROUP_DATUMS} datums must be touched once; got {seen:?}"
    );
}

#[test]
fn elementwise_multiply_matches_the_specifications_fma_model() {
    let values = operands();
    let descriptor = flat_descriptor(DATUMS);
    let staged = staged_values(descriptor, &values);

    in_device(|dev| {
        let packed = run_kernel(dev, descriptor, &staged, &binary_kernel(sfpu::mul));
        let checked = check_against_model(&packed, &values, numerics::mul_bh);
        assert_eq!(
            checked, 60,
            "the comparison must cover every datum of the group but the four the \
             unpacker cannot reach"
        );
    });
}

#[test]
fn elementwise_add_matches_the_specifications_fma_model() {
    let values = operands();
    let descriptor = flat_descriptor(DATUMS);
    let staged = staged_values(descriptor, &values);

    in_device(|dev| {
        let packed = run_kernel(dev, descriptor, &staged, &binary_kernel(sfpu::add));
        let checked = check_against_model(&packed, &values, numerics::add_bh);
        assert_eq!(checked, 60);
    });
}

#[test]
fn elementwise_subtract_matches_the_specifications_fma_model() {
    let values = operands();
    let descriptor = flat_descriptor(DATUMS);
    let staged = staged_values(descriptor, &values);

    in_device(|dev| {
        let packed = run_kernel(dev, descriptor, &staged, &binary_kernel(sfpu::sub));
        let checked = check_against_model(&packed, &values, |a, b| {
            numerics::add_bh(a, b ^ 0x8000_0000)
        });
        assert_eq!(checked, 60);
    });
}

/// The kernel computes rather than copying.
///
/// The gates above would pass on a datapath that ignored the SFPU entirely if the
/// operands happened to make the result equal to one of them. This asserts the
/// results are neither operand, for enough of them that coincidence is not the
/// explanation.
#[test]
fn the_result_is_neither_operand() {
    let values = operands();
    let descriptor = flat_descriptor(DATUMS);
    let staged = staged_values(descriptor, &values);

    in_device(|dev| {
        let packed = run_kernel(dev, descriptor, &staged, &binary_kernel(sfpu::mul));
        let mut distinct = 0;
        let mut total = 0;
        for (flat, &got) in packed.iter().enumerate().take(GROUP_DATUMS).skip(4) {
            let (Some(a), Some(b)) = (
                datum_at_dst_flat(flat),
                datum_at_dst_flat(flat + GROUP_DATUMS),
            ) else {
                continue;
            };
            if b >= values.len() {
                continue;
            }
            total += 1;
            if got != values[a].to_bits() && got != values[b].to_bits() {
                distinct += 1;
            }
        }
        assert!(
            distinct * 10 >= total * 9,
            "only {distinct} of {total} results differ from both operands; the \
             kernel may not be computing at all"
        );
    });
}

/// Denormals flush and NaN canonicalises, and the model predicts both.
///
/// `docs/ttsim-divergence.md` row D records these as SFPU behaviour, found by
/// differential testing. What is new here is that the *oracle* predicts them: if
/// `fma_bh` did not flush denormals this gate would fail even though the hardware
/// was right, so it checks the pair rather than the device alone.
#[test]
fn denormals_and_nans_follow_the_model_through_the_whole_datapath() {
    let awkward = [
        0x0000_0001u32, // smallest denormal
        0x007f_ffff,    // largest denormal
        0x7fc0_0000,    // canonical NaN
        0x7f80_0001,    // signalling NaN
        0x7f80_0000,    // +inf
        0xff80_0000,    // -inf
        0x8000_0000,    // -0
        0x0000_0000,    // +0
        0x3f80_0000,    // 1.0
        0xbf80_0000,    // -1.0
        0x7f7f_ffff,    // FLT_MAX
        0x0080_0000,    // smallest normal
    ];
    let values: Vec<f32> = (0..2 * GROUP_DATUMS)
        .map(|i| f32::from_bits(awkward[(i * 7 + i / 5) % awkward.len()]))
        .collect();
    let descriptor = flat_descriptor(DATUMS);
    let staged = staged_values(descriptor, &values);

    in_device(|dev| {
        let packed = run_kernel(dev, descriptor, &staged, &binary_kernel(sfpu::mul));
        let checked = check_against_model(&packed, &values, numerics::mul_bh);
        assert_eq!(checked, 60);

        // And that the awkward cases actually occurred, so the gate is not passing
        // on ordinary arithmetic that happens to be in the list.
        let saw_nan = packed[4..GROUP_DATUMS].contains(&0x7fc0_0000);
        let saw_zero = packed[4..GROUP_DATUMS]
            .iter()
            .any(|&w| w & 0x7fff_ffff == 0);
        assert!(
            saw_nan,
            "no NaN result; the awkward operands did not reach the device"
        );
        assert!(
            saw_zero,
            "no zero result; denormal flushing was never exercised"
        );
    });
}

/// The `STALLWAIT` between the SFPU and the packer is emitted, in the right order.
///
/// Its *effect* is not observable on ttsim -- divergence row 34 records the same
/// for the packer's own wait -- so this asserts the instruction is in the stream
/// rather than that removing it breaks anything. A weak gate, and labelled as one:
/// the real check is on silicon.
#[test]
fn the_kernel_waits_for_the_sfpu_before_packing() {
    let program =
        eltwise_support::kernel_program(flat_descriptor(DATUMS), DATUMS, &binary_kernel(sfpu::mul));
    let pos = |w: Instruction| program.iter().position(|i| i.word() == w.word());
    let sfpu_at = pos(backend::wait_for_sfpu().unwrap()).expect("the SFPU wait must be present");
    let packer_at =
        pos(backend::wait_for_packer().unwrap()).expect("the packer wait must be present");
    let unpacker_at =
        pos(backend::wait_for_unpacker0().unwrap()).expect("the unpacker wait must be present");
    assert!(
        unpacker_at < sfpu_at && sfpu_at < packer_at,
        "the waits must appear in datapath order: unpacker, then SFPU, then packer"
    );
}

/// The tensor-level oracle: `burn-flex` decides which elements pair with which.
///
/// Two oracles, two distinct claims. `fma_bh` says what a *pair* of datums
/// produces, bit for bit, and is what every gate above asserts against. Burn says
/// which pairs there should be — the shape, the ordering, the composition — which
/// is the part a bit-exact scalar model cannot check and the part that will matter
/// once `tt_layout::Layout` drives a real tiled tensor rather than a flat run.
///
/// The two are made to agree deliberately: the operands are small integers, exactly
/// representable in FP32 and with exactly-representable products, so Burn's IEEE
/// arithmetic and Blackhole's FMA cannot differ. That is the point. Where they
/// *would* differ — denormals, NaN, the overflow rule — Burn is the wrong oracle
/// and `denormals_and_nans_follow_the_model_through_the_whole_datapath` is the
/// right one. Using a tolerance to paper over the gap is exactly what the
/// tolerance policy forbids.
///
/// **`burn-flex`, not `burn-ndarray`**: crates.io marks the latter
/// `[Deprecated] … use burn-flex, burn-cuda, burn-rocm`.
#[test]
fn the_element_pairing_matches_burn() {
    use burn_tensor::{Tensor, TensorData};

    // In Burn 0.21 the associated `Device` lives on `BackendTypes`, not on
    // `Backend` — one of the restructurings `docs/RUST_IMPL_PLAN.md`'s Phase 7
    // supertrait list predates. Naming the concrete device type sidesteps it.
    type B = burn_flex::Flex;
    let device = burn_flex::FlexDevice;

    // Small integers: exact in FP32, and their products are too.
    let values: Vec<f32> = (0..2 * GROUP_DATUMS)
        .map(|i| (i % 17) as f32 - 8.0)
        .collect();
    let descriptor = flat_descriptor(DATUMS);
    let staged = staged_values(descriptor, &values);

    // Burn computes the whole elementwise product of the two halves, laid out the
    // way the `Dst` lane mapping says the device pairs them.
    let a: Vec<f32> = (0..GROUP_DATUMS)
        .map(|flat| datum_at_dst_flat(flat).map_or(0.0, |i| values[i]))
        .collect();
    let b: Vec<f32> = (0..GROUP_DATUMS)
        .map(|flat| values[datum_at_dst_flat(flat + GROUP_DATUMS).unwrap()])
        .collect();
    let ta = Tensor::<B, 1>::from_data(TensorData::new(a, [GROUP_DATUMS]), &device);
    let tb = Tensor::<B, 1>::from_data(TensorData::new(b, [GROUP_DATUMS]), &device);
    let expected: Vec<f32> = (ta * tb)
        .into_data()
        .to_vec()
        .expect("burn should give f32 back");

    in_device(|dev| {
        let packed = run_kernel(dev, descriptor, &staged, &binary_kernel(sfpu::mul));
        let mut checked = 0;
        for (flat, (&got, want)) in packed
            .iter()
            .zip(expected.iter())
            .enumerate()
            .take(GROUP_DATUMS)
            .skip(4)
        {
            assert_eq!(
                got,
                want.to_bits(),
                "Dst[{}][{}]: device {got:#010x}, burn {:#010x}",
                flat / 16,
                flat % 16,
                want.to_bits()
            );
            checked += 1;
        }
        assert_eq!(checked, 60, "the comparison must not be vacuous");
    });
}
