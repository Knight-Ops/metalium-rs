//! The instruction corpus: encode a program on the host, run it, check every datum.
//!
//! Step 4 proved one instruction stream produces one answer. This is the general
//! form: the host encodes a program with `tt_isa::isa`, the generic `corpus`
//! firmware pushes it and copies `Dst` back, and the host checks the result against
//! a reimplementation of the page's functional model. Adding an instruction is a
//! case in this file, not a firmware change.
//!
//! # What this can and cannot cover
//!
//! Encoding is checked exhaustively elsewhere — `cargo xtask gen-isa` compares every
//! one of the 148 encodings against the specification's two independent descriptions
//! of it, and `tt_isa::isa`'s tests check the table's structure. That is a different
//! claim from *this* one, which is that ttsim executes the encoding and agrees with
//! the documented model.
//!
//! Most instructions cannot be run in isolation yet: `UNPACR` needs configured
//! unpackers, `MVMUL` needs valid `SrcA`/`SrcB`. What is here is the part of the
//! Vector Unit that touches only `LReg` and `Dst`, which needs no surrounding state
//! at all. The rest arrives with the units that feed it, in Phases 5 and 6.
//!
//! # Every case is a whole-process risk
//!
//! ttsim `_Exit`s on anything it does not model, so each run happens inside its own
//! `fork_scope` and an instruction it refuses becomes a recorded result rather than
//! a dead test runner. That makes this a discovery tool: see
//! [`instructions_ttsim_declines_to_execute`].

use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::sfpu::{self, loadi_mode, store_format};
use tt_tests::harness::{self, in_device, Dev, Run, SENTINEL};

/// Run `program` and return the first `rows` rows of `Dst`, 16 datums each.
///
/// The shared runner, plus the one assertion every corpus case wants: that the
/// firmware wrote *something*, so a row it never touched is not mistaken for a
/// computed zero.
fn run(dev: &mut Dev<'_>, program: &[Instruction], rows: u32) -> Vec<u32> {
    let out = harness::run(dev, &Run::new(program).dump_rows(rows)).dst;
    assert!(
        out.iter().any(|&v| v != SENTINEL),
        "the firmware wrote nothing into Dst"
    );
    out
}

/// `SFPLOADI` an FP32 constant into `LReg[vd]`, as two halves.
fn load(vd: u32, value: f32) -> [Instruction; 2] {
    sfpu::load_f32(vd, value.to_bits()).unwrap()
}

/// Store `LReg[vd]` to `Dst` rows 0..=3 and read back lane 0, which lands at
/// `Dst[0][0]` (`SFPSTORE.md`: `Row = (Addr & ~3) + Lane / 8`,
/// `Column = (Lane & 7) * 2`).
fn store_fp32(vd: u32) -> Instruction {
    sfpu::store(vd, store_format::FP32, 0, 0).unwrap()
}

/// Build `load a -> LReg[0]`, `load b -> LReg[1]`, `op`, `store LReg[2]`.
fn binary_program(a: f32, b: f32, op: Instruction) -> Vec<Instruction> {
    let mut p = Vec::new();
    p.extend(load(0, a));
    p.extend(load(1, b));
    p.push(op);
    p.push(store_fp32(2));
    p
}

/// Lane 0 of the store, which is `Dst[0][0]` — the first datum of the dump.
fn lane0(dump: &[u32]) -> u32 {
    dump[0]
}

#[test]
fn the_generic_runner_reproduces_the_step_four_answer() {
    // The same computation step 4 makes, expressed as host-side data rather than as
    // firmware. If this disagrees with `step4_tensix.rs`, the runner is wrong.
    in_device(|dev| {
        let program = binary_program(3.0, 2.0, sfpu::mul(0, 1, 2).unwrap());
        let dump = run(dev, &program, 4);
        assert_eq!(lane0(&dump), 0x40C0_0000, "3.0 * 2.0");
    });
}

#[test]
fn sfpstore_writes_the_lanes_the_functional_model_says_it_does() {
    // One `SFPSTORE` covers 32 lanes across four rows, even columns only. The odd
    // columns must be untouched -- if they were written, a later test reading
    // column 0 could be reading a neighbour's result.
    in_device(|dev| {
        let mut p = Vec::new();
        p.extend(load(0, 1.5));
        p.push(store_fp32(0));
        let dump = run(dev, &p, 4);

        for row in 0..4usize {
            for col in 0..16usize {
                let value = dump[row * 16 + col];
                if col % 2 == 0 {
                    assert_eq!(
                        value,
                        1.5f32.to_bits(),
                        "Dst[{row}][{col}] should hold the stored datum"
                    );
                } else {
                    // Not the sentinel: the firmware copies every column of every
                    // dumped row, so an odd column holds whatever `Dst` holds --
                    // which `Dst.md` says is `UnpredictableValue()` before a scrub.
                    // What matters is that the store did not reach it.
                    assert_ne!(
                        value,
                        1.5f32.to_bits(),
                        "Dst[{row}][{col}] is an odd column and the store must not \
                         have reached it"
                    );
                }
            }
        }
    });
}

#[test]
fn sfpmad_matches_the_hosts_fused_multiply_add() {
    // A genuine three-operand MAD, which the step 4 firmware never exercises: it
    // only ever uses the SFPMUL spelling with the constant-zero register.
    in_device(|dev| {
        for (a, b, c) in [
            (3.0f32, 2.0f32, 1.0f32),
            (-1.5, 4.0, 0.25),
            (0.5, 0.5, -0.125),
            (7.0, -3.0, 10.0),
        ] {
            let mut p = Vec::new();
            p.extend(load(0, a));
            p.extend(load(1, b));
            p.extend(load(2, c));
            p.push(sfpu::mad(0, 1, 2, 3, 0).unwrap());
            p.push(store_fp32(3));
            let dump = run(dev, &p, 4);
            assert_eq!(lane0(&dump), (a * b + c).to_bits(), "{a} * {b} + {c}");
        }
    });
}

#[test]
fn sfpmov_and_sfpabs_agree_with_the_host() {
    in_device(|dev| {
        // SFPMOV with Mod1 = 0 is a plain copy of LReg[VC] to LReg[VD].
        let mut p = Vec::new();
        p.extend(load(1, -2.75));
        p.push(encode::sfpmov(1, 2, 0).unwrap());
        p.push(store_fp32(2));
        assert_eq!(lane0(&run(dev, &p, 4)), (-2.75f32).to_bits(), "SFPMOV");

        // SFPABS with Mod1 = 1 is a floating-point absolute value: clear the sign.
        let mut p = Vec::new();
        p.extend(load(1, -2.75));
        p.push(encode::sfpabs(1, 2, 1).unwrap());
        p.push(store_fp32(2));
        assert_eq!(lane0(&run(dev, &p, 4)), 2.75f32.to_bits(), "SFPABS");
    });
}

/// The divergence log records that the SFPU drops the sign of a zero result unless
/// the addend is `-0`, which is why `sfpu::mul` sets `NEGATE_VC`. This is that
/// claim as an executable test rather than a note.
#[test]
fn negative_zero_survives_a_multiply_only_because_the_addend_is_negative() {
    in_device(|dev| {
        let with_fix = binary_program(-1.0, 0.0, sfpu::mul(0, 1, 2).unwrap());
        assert_eq!(
            lane0(&run(dev, &with_fix, 4)),
            (-0.0f32).to_bits(),
            "sfpu::mul sets NEGATE_VC, so the addend is -0 and the sign survives"
        );

        // The same multiply with a `+0` addend, which is what a naive SFPMUL does.
        let naive = binary_program(-1.0, 0.0, sfpu::mad(0, 1, sfpu::LREG_ZERO, 2, 0).unwrap());
        assert_eq!(
            lane0(&run(dev, &naive, 4)),
            0.0f32.to_bits(),
            "without NEGATE_VC the sign is lost -- this is the bug the wrapper avoids"
        );
    });
}

/// `docs/ttsim-divergence.md` entry D: denormal operands and results are flushed to
/// zero, and every NaN is canonicalised to `0x7FC0_0000`. It was recorded from
/// reading ttsim's source and noted as not yet exercised by a gate. It is now.
#[test]
fn denormals_flush_and_nans_canonicalise() {
    in_device(|dev| {
        // A denormal operand. The host would produce a denormal product; the SFPU
        // flushes it.
        let denormal = f32::from_bits(0x0000_0001);
        let p = binary_program(denormal, 1.0, sfpu::mul(0, 1, 2).unwrap());
        let got = lane0(&run(dev, &p, 4));
        assert_eq!(got, 0, "a denormal operand flushes to +0");
        assert_ne!(
            got,
            (denormal * 1.0).to_bits(),
            "the host keeps the denormal, so this is a real divergence"
        );

        // Every NaN, whatever its payload, comes back canonical.
        let payload_nan = f32::from_bits(0x7F80_1234);
        let p = binary_program(payload_nan, 1.0, sfpu::mul(0, 1, 2).unwrap());
        assert_eq!(
            lane0(&run(dev, &p, 4)),
            0x7FC0_0000,
            "NaN is canonicalised, payload and all"
        );
    });
}

/// `SFPLOADI`'s data-type modes, against the conversions `SFPLOADI.md` documents.
#[test]
fn sfploadi_modes_convert_the_way_the_page_says() {
    in_device(|dev| {
        // FLOATB: the immediate is BF16, widened by `imm << 16`.
        let p = vec![
            sfpu::loadi(0, loadi_mode::FLOATB, 0x40C0).unwrap(),
            store_fp32(0),
        ];
        assert_eq!(lane0(&run(dev, &p, 4)), 0x40C0_0000, "FLOATB widens by 16");

        // UPPER then LOWER builds an arbitrary FP32, which is what `load_f32` does.
        let mut p = Vec::new();
        p.extend(load(0, core::f32::consts::PI));
        p.push(store_fp32(0));
        assert_eq!(
            lane0(&run(dev, &p, 4)),
            core::f32::consts::PI.to_bits(),
            "UPPER + LOWER reconstructs a value with a non-zero low half"
        );

        // USHORT loads an integer, and storing it as FP32 is not a reinterpretation
        // -- 0x0000_BEEF read as FP32 is a denormal, and the SFPU flushes denormals
        // to zero. The integer store path needs `SFPSTORE_MOD0_INT32` and the
        // sign/magnitude conversion `Dst.md` documents, which is its own piece of
        // work; this pins the trap so nobody rediscovers it by debugging a zero.
        let p = vec![
            sfpu::loadi(0, loadi_mode::USHORT, 0xBEEF).unwrap(),
            store_fp32(0),
        ];
        assert_eq!(
            lane0(&run(dev, &p, 4)),
            0,
            "an integer stored through the FP32 path is a denormal, and flushes"
        );
    });
}

/// What the simulator declines to execute is a finding, not a blocked test.
///
/// Each of these runs in its own fork, so an instruction ttsim does not model
/// terminates that child and is recorded here rather than killing the run. An
/// instruction that starts passing is as much a finding as one that stops: it means
/// the simulator gained a model, and the corpus can grow.
#[cfg(not(feature = "silicon"))]
#[test]
fn instructions_ttsim_declines_to_execute() {
    /// Does this program run to completion under the simulator?
    fn survives(program: &[Instruction]) -> bool {
        harness::survives(|dev| {
            let _ = run(dev, program, 4);
        })
    }

    // `SFPLOADMACRO` is unsupported in ttsim's SFPU, stated in its own README and
    // recorded as divergence 7. This is the first time the repository has actually
    // watched it refuse, rather than taking the README's word for it.
    let mut p = Vec::new();
    p.extend(load(0, 1.0));
    p.push(encode::Sfploadmacro::ZERO.encode().unwrap());
    p.push(store_fp32(0));
    assert!(
        !survives(&p),
        "ttsim's README says SFPLOADMACRO is unsupported; if this now runs, the \
         simulator has gained a model and divergence 7 needs revisiting"
    );

    // The control: the same shape of program without it runs fine, so the failure
    // above is the instruction and not the harness.
    let mut p = Vec::new();
    p.extend(load(0, 1.0));
    p.push(sfpu::nop());
    p.push(store_fp32(0));
    assert!(survives(&p), "the control program must run");
}

/// What `SFPLOADMACRO` does on silicon, where ttsim cannot say (divergence row 7).
///
/// A smoke test, not a semantic one: with no macro configured, the pages specify
/// nothing this could assert about the result. What it does establish is that
/// the instruction retires and the core reaches `DONE` -- that an unconfigured
/// macro neither hangs the coprocessor nor wedges the pushing core -- and it
/// prints what `Dst` holds so the first run is a measurement.
#[cfg(feature = "silicon")]
#[test]
fn sfploadmacro_retires_on_silicon() {
    tt_tests::harness::assert_on_silicon();
    in_device(|dev| {
        let mut p = Vec::new();
        p.extend(load(0, 1.0));
        p.push(encode::Sfploadmacro::ZERO.encode().unwrap());
        p.push(store_fp32(0));
        let dst = run(dev, &p, 4);
        println!(
            "SFPLOADMACRO then SFPSTORE LReg[0]: Dst row 0 = {:08x?}",
            &dst[..16]
        );
    });
}
