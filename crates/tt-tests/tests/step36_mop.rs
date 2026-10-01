//! Phase 10 gate (X2): the MOP Expander.
//!
//! A role's `MopCfg` comes from its mailbox (`tt_isa::mailbox::MOP_CFG`): the
//! runner waits for the expander to be idle and writes the nine words before
//! pushing. Each case loads a configuration whose slots are integer adds of
//! distinct amounts to one SFPU register (`SFPIADD` by an immediate), issues
//! one `MOP` (two for template 0: the `MOP_CFG` that sets the mask's high
//! half, then the `MOP`), and stores the register over a whole tile -- so the
//! stored integer is a sum that says how many times each slot ran. The claim:
//! the device's tile **bit for bit** the interpreter's, running the same
//! program with the `MOP` replaced by the page's functional model's expansion
//! (`tt_isa::frontend::mop::expand`); and the first case's sum is also the
//! number worked out by hand, so the model cannot agree with the device by
//! being wrong the same way.
//!
//! `MOP` and `MOP_CFG` are drawn only on Wormhole pages: this gate is what
//! confirms their layouts on Blackhole (every field set to more than one
//! value: both templates, a `Count1` of 19, a mask with bits in both halves),
//! on ttsim and then on silicon in its own run on a healthy tile, and the
//! generated table cites it (`xtask/src/gen_isa/measured.rs`, `CONFIRMED`).

use tt_isa::backend::{self, Before};
use tt_isa::frontend::mop::{self, MopConfig, Template0, Template1};
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_kernels::datapath::{pack_tile_from_dst, state_id, thread_config, OUT};
use tt_kernels::sfpu::interp::Vector;
use tt_kernels::sfpu::{Format, LReg, LoopPolicy, Program};
use tt_tests::harness::{self, Roles, Run};

/// `L2 += n`, as an integer.
fn add(n: i32) -> Instruction {
    encode::sfpiadd(n as u32 & 0xfff, 2, 2, 1 | 4).unwrap()
}

/// `L2 = 0`, then `mops` (or, for the model, their expansion), then `L2`
/// stored as an integer over `Dst` rows 128..192.
fn program(mops: &[Instruction]) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.loadi_bits(LReg::L2, 0);
    for &i in mops {
        p.raw(i);
    }
    p.for_each_row_group(64, |p, o| p.store(LReg::L2, Format::Int32, 128 + o));
    p.finish()
}

fn on_device(dev: &mut harness::Dev<'_>, math: &[Instruction], cfg: MopConfig) -> Vec<u32> {
    let unpack = thread_config();
    let mut m = vec![state_id()];
    m.extend_from_slice(math);
    m.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
    let mut pack = vec![state_id()];
    pack.extend(pack_tile_from_dst(OUT, 128));
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    let sentinel = vec![0xA5u8; 4096];
    let out = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: &unpack,
            math: &m,
            pack: &pack,
        })
        .mop([None, Some(cfg), None])
        .stage(&[(OUT, &sentinel)])
        .dump_rows(0)
        .read_back(&[(OUT, 4096)]),
    );
    out.l1[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn model(mops: &[Instruction], cfg: &MopConfig) -> Vec<u32> {
    let mut expanded = Vec::new();
    let mut mask_hi = 0;
    for &i in mops {
        if core::ptr::eq(i.def(), &tt_isa::isa::generated::defs::MOP_CFG) {
            mask_hi = i.operand("MaskHi").unwrap();
        } else {
            mop::expand(cfg, i, mask_hi, &mut expanded);
        }
    }
    let mut v = Vector::new();
    v.run(&program(&expanded)).unwrap();
    v.tile(128)
}

fn cases() -> Vec<(&'static str, MopConfig, Vec<Instruction>, Option<u32>)> {
    let t1 = Template1 {
        outer: 3,
        inner: 4,
        start: Some(add(1)),
        end: Some((add(2), Some(add(3)))),
        loop_op: add(10),
        loop_op1: None,
        last_of_last: add(1000),
        last: add(100),
    };
    // Per outer pass: 1 + three loop ops of 10 + its last op + 2 + 3; the last
    // op is 100 on the first two passes and 1000 on the third.
    let by_hand = 2 * (1 + 30 + 100 + 5) + (1 + 30 + 1000 + 5);
    let alternating = Template1 {
        outer: 2,
        inner: 3,
        start: None,
        end: None,
        loop_op: add(20),
        loop_op1: Some(add(300)),
        last_of_last: add(7),
        last: add(5),
    };
    let t0 = Template0 {
        a0: add(1),
        a123: Some([add(2), add(4), add(8)]),
        b: Some(add(16)),
        skip_a0: add(100),
        skip_b: add(200),
    };
    vec![
        (
            "template 1: start, loop, last ops, two end ops",
            MopConfig::Template1(t1),
            vec![mop::mop_template1()],
            Some(by_hand as u32),
        ),
        (
            "template 1: alternating loop ops",
            MopConfig::Template1(alternating),
            vec![mop::mop_template1()],
            None,
        ),
        (
            "template 0: a mask across both halves",
            MopConfig::Template0(t0),
            mop::mop_template0(20, 0b1010_0000_0000_0000_0110)
                .unwrap()
                .to_vec(),
            None,
        ),
    ]
}

#[test]
fn a_mop_expands_as_the_page_models_it() {
    harness::in_device(|dev| {
        for (name, cfg, mops, by_hand) in cases() {
            let want = model(&mops, &cfg);
            if let Some(n) = by_hand {
                assert!(
                    want.iter().all(|&w| w == n),
                    "{name}: the model's sum is {}, by hand {n}",
                    want[0]
                );
            }
            assert!(
                want.iter().all(|&w| w == want[0]),
                "{name}: every lane the same"
            );
            let got = on_device(dev, &program(&mops), cfg);
            assert!(
                got == want,
                "{name}: device {:#x} (first datum), model {:#x}",
                got[0],
                want[0]
            );
        }
    });
}

/// The configuration is what the runner loads, not leftover state: the same
/// `MOP` under two configurations gives two sums.
#[test]
fn each_run_loads_its_own_configuration() {
    let t = |outer| {
        MopConfig::Template1(Template1 {
            outer,
            inner: 2,
            start: None,
            end: None,
            loop_op: add(1),
            loop_op1: None,
            last_of_last: add(1),
            last: add(1),
        })
    };
    harness::in_device(|dev| {
        let math = program(&[mop::mop_template1()]);
        let five = on_device(dev, &math, t(5));
        let two = on_device(dev, &math, t(2));
        assert_eq!((five[0], two[0]), (10, 4));
    });
}
