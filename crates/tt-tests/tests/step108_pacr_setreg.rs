//! `PACR_SETREG` and `UNPACR_NOP_SETREG`: research closure, simulator refusal only.
//!
//! Both are `[-]` (see `docs/plans/hardware-coverage-closeout.md`, lane F). The
//! address of their write is `SetRegBase[AddrSel] + (AddrMid << 12)`, and
//! `SetRegBase`/`SetRegHiScaler` are TDMA-RISC state that only the Wormhole
//! `TDMA-RISC.md` page documents (`0xFFB1_1038` -> `SetPackRegAddr`,
//! `0xFFB1_1028`/`0xFFB1_103C` -> `SetScaler`). The Blackhole pin has no
//! `TDMA-RISC.md`, no `PACR_SETREG.md` and no `UNPACR_NOP_SETREG.md`, so there is
//! no pinned way to initialise that state and no typed API is offered. These gates
//! only record that ttsim refuses both instructions, with a surviving control on
//! the same role and drain, so a later pin or simulator change is noticed.
#![cfg(not(feature = "silicon"))]
use tt_isa::{
    backend::{self, Before},
    isa::{generated::encode, Instruction},
};
use tt_tests::harness::{self, Roles, Run};

/// Run `p` on one role (0 unpack, 1 math, 2 pack); `false` means ttsim refused.
fn survives_on(role: usize, p: &[Instruction]) -> bool {
    harness::survives(|dev| {
        let mut roles = [&[][..]; 3];
        roles[role] = p;
        harness::run(
            dev,
            &Run::roles(Roles {
                unpack: roles[0],
                math: roles[1],
                pack: roles[2],
            })
            .dump_rows(0),
        );
    })
}
fn drain() -> Instruction {
    backend::wait_for_scalar(Before::EVERYTHING).unwrap()
}

#[test]
fn simulator_refuses_pacr_setreg_and_unpacr_nop_setreg() {
    // Controls: the same role and drain without the instruction survive.
    assert!(survives_on(2, &[drain()]));
    assert!(survives_on(0, &[drain()]));

    // `tensix_decode_pacr_setreg` (UnsupportedFunctionality on ttsim). AddrMid and
    // AddrSel are zero: nothing here names an address.
    let pack = encode::pacr_setreg(0, 1, 0).unwrap();
    assert_eq!(pack.def().mnemonic(), "PACR_SETREG");
    assert!(!survives_on(2, &[pack, drain()]));

    // `UNPACR_NOP_SETREG` is documented `UnsupportedFunctionality` even on Wormhole.
    let unpack = encode::UnpacrNopSetreg::ZERO.value11(1).encode().unwrap();
    assert_eq!(unpack.def().key(), "UNPACR_NOP_SETREG");
    assert!(!survives_on(0, &[unpack, drain()]));
}
