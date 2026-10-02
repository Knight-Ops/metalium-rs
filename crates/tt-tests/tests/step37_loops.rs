//! Phase 10 gate (X2b): loops lowered to the frontend's expanders
//! (`tt_kernels::loops`).
//!
//! A math program of loops whose bodies are integer adds of distinct amounts
//! to one SFPU register, then the register stored over a tile: the stored sum
//! says how many times each instruction ran. Lowered -- one loop under the
//! thread's MOP configuration, a `MOP` looping a `REPLAY` of its recorded
//! body, the others plain `REPLAY`s -- the device must store exactly what the
//! unrolled program stores, and the lowered program must be the fewer words.
//! The planner's own claim, that the frontend's output is the unrolled stream
//! word for word, is its unit tests'; this is the hardware agreeing with the
//! model the claim is made against.

use tt_isa::backend::{self, Before};
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_kernels::datapath::{pack_tile_from_dst, state_id, thread_config, OUT};
use tt_kernels::loops::{lower, Item, LoopForm};
use tt_kernels::sfpu::{Format, LReg, LoopPolicy, Program};
use tt_tests::harness::{self, Roles, Run};

fn add(n: i32) -> Instruction {
    encode::sfpiadd(n as u32 & 0xfff, 2, 2, 1 | 4).unwrap()
}

/// `L2 = 0`, `middle`, then `L2` stored as an integer over rows 128..192 --
/// the store unrolled, so the only `REPLAY`s are the planner's.
fn math(middle: &[Instruction]) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.loadi_bits(LReg::L2, 0);
    for &i in middle {
        p.raw(i);
    }
    p.for_each_row_group(64, |p, o| p.store(LReg::L2, Format::Int32, 128 + o));
    p.finish()
}

fn on_device(
    dev: &mut harness::Dev<'_>,
    math: &[Instruction],
    mop: Option<tt_isa::frontend::mop::MopConfig>,
) -> Vec<u32> {
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
        .mop([None, mop, None])
        .stage(&[(OUT, &sentinel)])
        .dump_rows(0)
        .read_back(&[(OUT, 4096)]),
    );
    out.l1[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn lowered_loops_store_what_the_unrolled_program_stores() {
    let body =
        |base: i32, n: i32| -> Vec<Item> { (0..n).map(|k| Item::I(add(base + k))).collect() };
    let items = vec![
        Item::I(add(1)),
        // 70 iterations of a 3-add body: the MOP's (three MOPs, 32 + 32 + 5).
        Item::Repeat {
            times: 70,
            body: body(10, 3),
        },
        // A nested loop flattened into one recorded body, replayed.
        Item::Repeat {
            times: 6,
            body: vec![
                Item::I(add(100)),
                Item::Repeat {
                    times: 2,
                    body: body(200, 2),
                },
            ],
        },
        Item::I(add(3)),
    ];
    let lowered = lower(&items);
    assert!(
        matches!(lowered.loops[0], LoopForm::Mop { mops: 3, .. })
            && matches!(lowered.loops[1], LoopForm::Replayed { .. }),
        "{:?}",
        lowered.loops
    );
    let unrolled = Item::unrolled(&items);
    assert!(
        lowered.words.len() * 4 < unrolled.len(),
        "{} words against {}",
        lowered.words.len(),
        unrolled.len()
    );
    // By hand: 1 + 70 * (10 + 11 + 12) + 6 * (100 + 2 * (200 + 201)) + 3.
    let by_hand = 1 + 70 * 33 + 6 * (100 + 2 * 401) + 3;
    harness::in_device(|dev| {
        let want = on_device(dev, &math(&unrolled), None);
        assert!(
            want.iter().all(|&w| w == by_hand),
            "unrolled: {} by hand {by_hand}",
            want[0]
        );
        let got = on_device(dev, &math(&lowered.words), lowered.mop);
        assert!(
            got == want,
            "lowered {:#x}, unrolled {:#x}",
            got[0],
            want[0]
        );
    });
}
