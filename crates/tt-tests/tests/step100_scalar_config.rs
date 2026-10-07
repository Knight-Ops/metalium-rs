//! Scalar/configuration foundation; independent models from pinned instruction
//! pseudocode. Diagnostic descriptors never reach a memory engine.
use tt_isa::{
    backend::{self, ConfigOperation as Alu, MaskMode, Scratch},
    cfg::generated::{global, thcon, thread},
    isa::Instruction,
    scalar::{self, Bitwise, Comparison, Direction, Operand},
    tensix,
};
use tt_tests::harness::{self, Dev, Roles, Run};
const SLOTS: [u16; 3] = [
    global::SCRATCH_SEC0_val.addr32(),
    global::SCRATCH_SEC1_val.addr32(),
    global::SCRATCH_SEC2_val.addr32(),
];
const BASES: [u16; 2] = [
    thcon::THCON_SEC0_REG3_Base_address.addr32(),
    thcon::THCON_SEC1_REG3_Base_address.addr32(),
];
const ALUS: [Alu; 8] = [
    Alu::Or,
    Alu::And,
    Alu::Xor,
    Alu::Add,
    Alu::OrNot,
    Alu::AndNot,
    Alu::XorNot,
    Alu::Sub,
];
fn bank(id: u16) -> Instruction {
    backend::set_thread_entry(thread::CFG_STATE_ID_StateID.addr32(), id).unwrap()
}
fn write(p: &mut Vec<Instruction>, gpr: u32, addr: u16, value: u32) {
    p.extend(backend::set_gpr(gpr, value).unwrap());
    p.push(backend::write_word(gpr, addr).unwrap());
    p.push(backend::nop());
}
fn debug(dev: &mut Dev<'_>, addr: u16, bank: u16) -> u32 {
    let w = dev
        .alloc_window(tt_device::tlb::WindowKind::TwoMib)
        .unwrap();
    let tile = harness::tensix_tile();
    dev.write32(
        &w,
        tile,
        tensix::CFGREG_RD_CNTL,
        u32::from(addr) + u32::from(bank) * 224,
    )
    .unwrap();
    harness::advance(dev, 64);
    dev.read32(&w, tile, tensix::CFGREG_RDDATA).unwrap()
}
fn run_thread(dev: &mut Dev<'_>, p: &[Instruction], t: usize) {
    let mut roles = [&[][..]; 3];
    roles[t] = p;
    harness::run(
        dev,
        &Run::roles(Roles {
            unpack: roles[0],
            math: roles[1],
            pack: roles[2],
        })
        .dump_rows(0),
    );
}
// Independent pseudocode translation: no decoding of the tested instruction.
fn config_model(
    mut dst: u32,
    scratch: u32,
    op: Alu,
    mode: MaskMode,
    width: u32,
    rotation: u32,
) -> u32 {
    let mask = (u32::MAX >> (32 - width)).rotate_right(rotation);
    let v = (scratch & (u32::MAX >> (32 - width))).rotate_right(rotation);
    if mode == MaskMode::Replace {
        dst &= !mask;
    }
    match op {
        Alu::Or => dst | v,
        Alu::And => dst & v,
        Alu::Xor => dst ^ v,
        Alu::Add => dst.wrapping_add(v),
        Alu::OrNot => dst | !v,
        Alu::AndNot => dst & !v,
        Alu::XorNot => dst ^ !v,
        Alu::Sub => dst.wrapping_sub(v),
    }
}
#[test]
fn checked_encodings_and_boundaries() {
    for g in [0, 63] {
        for a in [0, 223] {
            let [read, wait] = backend::read_word(g, a).unwrap();
            assert_eq!(read.operand("ResultReg"), Some(g));
            assert_eq!(read.operand("CfgIndex"), Some(a.into()));
            assert_eq!(
                wait.operand("BlockMask"),
                Some(backend::Before::EVERYTHING.mask())
            );
            assert_eq!(
                wait.operand("ConditionMask"),
                Some(backend::cond::CONFIG_BUSY)
            );
        }
    }
    assert!(backend::read_word(64, 0).is_err());
    assert!(backend::read_word(0, 224).is_err());
    for target in SLOTS.into_iter().chain(BASES) {
        for op in ALUS {
            for mode in [MaskMode::Replace, MaskMode::Preserve] {
                for s in [
                    Scratch::Slot0,
                    Scratch::Slot1,
                    Scratch::Slot2,
                    Scratch::CurrentThread,
                ] {
                    for width in [1, 32] {
                        for rot in [0, 31] {
                            let [i, w] =
                                backend::modify_word(target, s, op, mode, width, rot).unwrap();
                            assert_eq!(i.operand("MaskWidth"), Some(width - 1));
                            assert_eq!(i.operand("RotateAmt"), Some(rot));
                            assert_eq!(i.operand("AluMode"), Some(op as u32));
                            assert_eq!(i.operand("ScratchIndex"), Some(s as u32));
                            assert_eq!(i.operand("MaskMode"), Some(mode as u32));
                            assert_eq!(
                                w.operand("ConditionMask"),
                                Some(backend::cond::CONFIG_BUSY)
                            );
                        }
                    }
                }
            }
        }
    }
    for (target, w, r) in [
        (0, 1, 0),
        (backend::STATE_RESET_EN_ADDR32, 1, 0),
        (SLOTS[0], 0, 0),
        (SLOTS[0], 33, 0),
        (SLOTS[0], 1, 32),
    ] {
        assert!(
            backend::modify_word(target, Scratch::Slot0, Alu::Or, MaskMode::Replace, w, r).is_err()
        );
    }
    for op in 0..10 {
        for right in [
            Operand::Register(0),
            Operand::Register(63),
            Operand::Immediate(0),
            Operand::Immediate(if op == 5 || op == 6 { 31 } else { 63 }),
        ] {
            for g in [0, 63] {
                assert!(scalar_instruction(op, g, g, right).is_ok());
            }
        }
        for g in [64, u32::MAX] {
            assert!(scalar_instruction(op, g, 0, Operand::Register(0)).is_err());
            assert!(scalar_instruction(op, 0, g, Operand::Register(0)).is_err());
            assert!(scalar_instruction(op, 0, 0, Operand::Register(g)).is_err());
        }
        assert!(scalar_instruction(
            op,
            0,
            0,
            Operand::Immediate(if op == 5 || op == 6 { 32 } else { 64 })
        )
        .is_err());
    }
}
fn scalar_instruction(
    op: usize,
    d: u32,
    l: u32,
    r: Operand,
) -> Result<Instruction, backend::EncodeError> {
    match op {
        0 => scalar::sub(d, l, r),
        1 => scalar::mul_u16(d, l, r),
        2..=4 => scalar::compare(
            d,
            l,
            r,
            [Comparison::Greater, Comparison::Less, Comparison::Equal][op - 2],
        ),
        5..=6 => scalar::shift(d, l, r, [Direction::Left, Direction::Right][op - 5]),
        7..=9 => scalar::bitwise(d, l, r, [Bitwise::And, Bitwise::Or, Bitwise::Xor][op - 7]),
        _ => unreachable!(),
    }
}
fn scalar_model(op: usize, l: u32, r: u32) -> u32 {
    match op {
        0 => l.wrapping_sub(r),
        1 => (l & 65535) * (r & 65535),
        2 => u32::from(l > r),
        3 => u32::from(l < r),
        4 => u32::from(l == r),
        5 => l << (r & 31),
        6 => l >> (r & 31),
        7 => l & r,
        8 => l | r,
        9 => l ^ r,
        _ => unreachable!(),
    }
}
#[test]
fn scalar_register_and_immediate_semantics() {
    harness::in_device(|dev| {
        for op in 0..10 {
            for immediate in [false, true] {
                if !cfg!(feature = "silicon") && (op != 1 || immediate) {
                    continue;
                }
                for (l, r) in [
                    (0, 0),
                    (0, 1),
                    (63, 63),
                    (0x80000000, 0x80000000),
                    (u32::MAX, 63),
                    (0x80000000, 31),
                    (0x1234ffff, 32),
                    (0xffff0001, 0xffff0002),
                    (1, 63),
                ] {
                    if immediate && r > if op == 5 || op == 6 { 31 } else { 63 } {
                        continue;
                    }
                    for (left, right, result) in [
                        (24, 25, 26),
                        (24, 28, 24),
                        (31, 24, 24),
                        (0, 63, 63),
                        (63, 0, 0),
                    ] {
                        let mut p = vec![bank(0)];
                        p.extend(backend::set_gpr(left, l).unwrap());
                        p.extend(backend::set_gpr(right, r).unwrap());
                        p.push(
                            scalar_instruction(
                                op,
                                result,
                                left,
                                if immediate {
                                    Operand::Immediate(r)
                                } else {
                                    Operand::Register(right)
                                },
                            )
                            .unwrap(),
                        );
                        p.push(backend::write_word(result, BASES[1]).unwrap());
                        p.push(backend::nop());
                        run_thread(dev, &p, 1);
                        assert_eq!(debug(dev,BASES[1],0),scalar_model(op,l,r),"op {op} immediate {immediate} l {l:x} r {r:x} registers {left}/{right}/{result}");
                    }
                }
            }
        }
    });
}
#[test]
#[cfg(feature = "silicon")]
fn configuration_banks_threads_selectors_and_modes() {
    harness::in_device(|dev| {
        for t in 0..3 {
            for b in 0..2 {
                for op in ALUS {
                    for mode in [MaskMode::Replace, MaskMode::Preserve] {
                        for sel in [
                            Scratch::Slot0,
                            Scratch::Slot1,
                            Scratch::Slot2,
                            Scratch::CurrentThread,
                        ] {
                            for width in [1, 32] {
                                for rot in [0, 31] {
                                    let source = [0x12345679, 0x87654321, 0xfffffffd];
                                    let dst = 0xfffffffe;
                                    let mut p = vec![bank(1 - b)];
                                    write(&mut p, 24, BASES[0], 0x13579bdf);
                                    p.push(bank(b));
                                    for (addr, v) in SLOTS.into_iter().zip(source) {
                                        write(&mut p, 24, addr, v);
                                    }
                                    write(&mut p, 24, BASES[0], dst);
                                    p.extend(
                                        backend::modify_word(BASES[0], sel, op, mode, width, rot)
                                            .unwrap(),
                                    );
                                    p.extend(backend::read_word(31, BASES[0]).unwrap());
                                    p.push(backend::write_word(31, SLOTS[2]).unwrap());
                                    p.push(backend::nop());
                                    run_thread(dev, &p, t);
                                    let selected = if sel == Scratch::CurrentThread {
                                        t
                                    } else {
                                        sel as usize
                                    };
                                    let expected =
                                        config_model(dst, source[selected], op, mode, width, rot);
                                    assert_eq!(debug(dev,BASES[0],b),expected,"thread {t} bank {b} {op:?} {mode:?} {sel:?} width {width} rot {rot}");
                                    assert_eq!(
                                        debug(dev, SLOTS[2], 1 - b),
                                        expected,
                                        "global scratch"
                                    );
                                    assert_eq!(
                                        debug(dev, BASES[0], 1 - b),
                                        0x13579bdf,
                                        "other bank preserved"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}

// Each invocation initializes GPRs 24–31 and all shared scratch words. A
// bounded tile offset is diagnostic data, never an address submitted to DMA.
#[cfg(feature = "silicon")]
fn descriptor(index: u32, stride: u32) -> (Vec<Instruction>, Vec<Instruction>, u32) {
    use tt_kernels::loops::{self, Item};
    let mut init = vec![bank(0)];
    for g in 24..32 {
        init.extend(backend::set_gpr(g, 0).unwrap());
    }
    for a in SLOTS {
        write(&mut init, 24, a, 0);
    }
    init.extend(backend::set_gpr(24, index).unwrap());
    init.extend(backend::set_gpr(25, stride).unwrap());
    let mut body = vec![
        scalar::bitwise(26, 24, Operand::Immediate(63), Bitwise::And).unwrap(),
        scalar::mul_u16(27, 26, Operand::Register(25)).unwrap(),
        scalar::shift(28, 27, Operand::Immediate(4), Direction::Left).unwrap(),
        scalar::sub(29, 28, Operand::Immediate(1)).unwrap(),
        scalar::compare(30, 26, Operand::Immediate(31), Comparison::Greater).unwrap(),
        backend::write_word(29, SLOTS[0]).unwrap(),
        backend::nop(),
    ];
    body.extend(
        backend::modify_word(SLOTS[1], Scratch::Slot0, Alu::Or, MaskMode::Replace, 32, 0).unwrap(),
    );
    body.extend(backend::read_word(31, SLOTS[1]).unwrap());
    body.push(scalar::bitwise(28, 31, Operand::Register(30), Bitwise::Xor).unwrap());
    // Establish the scalar-to-configuration publication boundary under replay.
    // Without it, card-0 run 1791342637 published the preceding shift result.
    body.push(
        backend::stallwait(
            backend::Before::EVERYTHING.mask(),
            backend::cond::CONFIG_BUSY,
        )
        .unwrap(),
    );
    body.push(backend::write_word(28, SLOTS[2]).unwrap());
    body.push(backend::nop());
    let items = [Item::Repeat {
        times: 3,
        body: body.into_iter().map(Item::I).collect(),
    }];
    let lowered = loops::lower_with(&items, false);
    assert!(
        lowered
            .words
            .iter()
            .any(|i| { i.def().mnemonic() == "REPLAY" && i.operand("Load") == Some(0) }),
        "the lowered consumer must execute a replay, beyond recording its body"
    );
    let mut unrolled = init.clone();
    unrolled.extend(Item::unrolled(&items));
    init.extend(lowered.words);
    let masked = index & 63;
    let offset = (masked * (stride & 65535)).wrapping_shl(4).wrapping_sub(1);
    (unrolled, init, offset ^ u32::from(masked > 31))
}
#[test]
#[cfg(feature = "silicon")]
fn replay_descriptor_changed_parameters_and_intervening_configuration() {
    harness::in_device(|dev| {
        for (index, stride) in [(7, 19), (127, 0x12340013), (0, 63), (32, 65535)] {
            let (unrolled, replay, want) = descriptor(index, stride);
            for (form, mut program) in [unrolled, replay].into_iter().enumerate() {
                program.push(backend::write_word(30, BASES[0]).unwrap());
                program.push(backend::nop());
                program.push(backend::write_word(26, BASES[1]).unwrap());
                program.push(backend::nop());
                let mut other = vec![bank(1)];
                for a in SLOTS.into_iter().chain(BASES) {
                    write(&mut other, 31, a, 0xa5a5a5a5);
                }
                run_thread(dev, &other, 2);
                run_thread(dev, &program, 0);
                assert_eq!(
                    debug(dev, BASES[1], 0),
                    index & 63,
                    "masked index form {form}"
                );
                assert_eq!(
                    debug(dev, BASES[0], 0),
                    u32::from((index & 63) > 31),
                    "compare index {index} form {form}"
                );
                assert_eq!(
                    debug(dev, SLOTS[2], 0),
                    want,
                    "index {index} stride {stride} form {form}"
                );
                assert_eq!(debug(dev, SLOTS[1], 1), want ^ u32::from((index & 63) > 31));
            }
        }
    });
}
#[test]
#[cfg(feature = "silicon")]
fn thread_local_registers_survive_other_threads() {
    harness::in_device(|dev| {
        let programs: Vec<_> = (0..3)
            .map(|t| {
                let mut p = vec![bank(0)];
                p.extend(backend::set_gpr(24, 0x12340000 + t).unwrap());
                p
            })
            .collect();
        harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &programs[0],
                math: &programs[1],
                pack: &programs[2],
            })
            .dump_rows(0),
        );
        let programs: Vec<_> = (0..3)
            .map(|t| {
                vec![
                    bank(0),
                    backend::write_word(24, SLOTS[t]).unwrap(),
                    backend::nop(),
                ]
            })
            .collect();
        harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &programs[0],
                math: &programs[1],
                pack: &programs[2],
            })
            .dump_rows(0),
        );
        for (t, a) in SLOTS.into_iter().enumerate() {
            assert_eq!(debug(dev, a, 0), 0x12340000 + t as u32);
        }
    });
}
#[test]
#[cfg(feature = "silicon")]
fn deterministic_negative_controls() {
    harness::in_device(|dev| {
        // Swapping inputs is observed on device; alternate hypotheses for width
        // and signedness are compared against the same independently read result.
        for (op, l, r) in [(0, 7, 19), (1, 0x12340003, 0x00020005), (2, 0x80000000, 1)] {
            let mut p = vec![bank(0)];
            p.extend(backend::set_gpr(24, l).unwrap());
            p.extend(backend::set_gpr(28, r).unwrap());
            p.push(
                scalar_instruction(
                    op,
                    31,
                    if op == 0 { 28 } else { 24 },
                    Operand::Register(if op == 0 { 24 } else { 28 }),
                )
                .unwrap(),
            );
            p.push(backend::write_word(31, SLOTS[2]).unwrap());
            p.push(backend::nop());
            run_thread(dev, &p, 1);
            let got = debug(dev, SLOTS[2], 0);
            if op == 0 {
                assert_ne!(got, scalar_model(op, l, r));
                assert_eq!(got, r.wrapping_sub(l));
            }
            if op == 1 {
                assert_eq!(got, scalar_model(op, l, r));
                assert_ne!(got, l.wrapping_mul(r));
            }
            if op == 2 {
                assert_eq!(got, scalar_model(op, l, r));
                assert_ne!(got, u32::from((l as i32) > (r as i32)));
            }
        }
        for (selector, rotation) in [
            (Scratch::Slot0, 0),
            (Scratch::Slot1, 0),
            (Scratch::Slot0, 31),
        ] {
            let mut p = vec![bank(0)];
            write(&mut p, 24, SLOTS[0], 1);
            write(&mut p, 24, SLOTS[1], 0);
            write(&mut p, 24, BASES[1], 0);
            p.extend(
                backend::modify_word(BASES[1], selector, Alu::Or, MaskMode::Replace, 1, rotation)
                    .unwrap(),
            );
            run_thread(dev, &p, 0);
            let got = debug(dev, BASES[1], 0);
            if selector == Scratch::Slot0 && rotation == 0 {
                assert_eq!(got, 1);
            } else {
                assert_ne!(got, 1);
            }
        }
    });
}

#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_form_probe() {
    for op in 0..10 {
        for immediate in [false, true] {
            let survives = harness::survives(|dev| {
                let mut p = vec![bank(0)];
                p.extend(backend::set_gpr(24, 7).unwrap());
                p.extend(backend::set_gpr(28, 3).unwrap());
                p.push(
                    scalar_instruction(
                        op,
                        31,
                        24,
                        if immediate {
                            Operand::Immediate(3)
                        } else {
                            Operand::Register(28)
                        },
                    )
                    .unwrap(),
                );
                p.push(backend::write_word(31, BASES[1]).unwrap());
                p.push(backend::nop());
                run_thread(dev, &p, 1);
                assert_eq!(debug(dev, BASES[1], 0), scalar_model(op, 7, 3));
            });
            assert_eq!(
                survives,
                op == 1 && !immediate,
                "op {op} immediate {immediate}"
            );
        }
    }
    for width in [1, 2, 8, 32] {
        for op in ALUS {
            let survives = harness::survives(|dev| {
                let mut p = vec![bank(0)];
                write(&mut p, 24, SLOTS[0], 3);
                write(&mut p, 24, BASES[0], 5);
                p.extend(
                    backend::modify_word(
                        BASES[0],
                        Scratch::Slot0,
                        op,
                        MaskMode::Preserve,
                        width,
                        0,
                    )
                    .unwrap(),
                );
                run_thread(dev, &p, 1);
                assert_eq!(
                    debug(dev, BASES[0], 0),
                    config_model(5, 3, op, MaskMode::Preserve, width, 0)
                );
            });
            assert_eq!(
                survives,
                width == 32 && op == Alu::Add,
                "{op:?} width {width}"
            );
        }
    }
    let survives = harness::survives(|dev| {
        let mut p = vec![bank(0)];
        write(&mut p, 24, BASES[0], 123);
        p.extend(backend::read_word(31, BASES[0]).unwrap());
        p.push(backend::write_word(31, BASES[1]).unwrap());
        p.push(backend::nop());
        run_thread(dev, &p, 1);
        assert_eq!(debug(dev, BASES[1], 0), 123);
    });
    assert!(survives, "RDCFG control must survive");
}

#[test]
fn configuration_read_banks_and_threads() {
    harness::in_device(|dev| {
        for t in 0..3 {
            for b in 0..2 {
                let mut p = vec![bank(1 - b)];
                write(&mut p, 24, BASES[0], 0x13579bdf);
                p.push(bank(b));
                write(&mut p, 24, BASES[0], 0x87654321);
                p.extend(backend::read_word(31, BASES[0]).unwrap());
                p.push(bank(1 - b));
                p.extend(backend::read_word(30, BASES[0]).unwrap());
                p.push(bank(0));
                p.push(backend::write_word(31, BASES[0]).unwrap());
                p.push(backend::nop());
                p.push(backend::write_word(30, BASES[1]).unwrap());
                p.push(backend::nop());
                run_thread(dev, &p, t);
                assert_eq!(debug(dev, BASES[0], 0), 0x87654321);
                assert_eq!(debug(dev, BASES[1], 0), 0x13579bdf);
            }
        }
    });
}
#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_config_variants_probe() {
    for sel in [
        Scratch::Slot0,
        Scratch::Slot1,
        Scratch::Slot2,
        Scratch::CurrentThread,
    ] {
        for mode in [MaskMode::Replace, MaskMode::Preserve] {
            for rot in [0, 31] {
                let survives = harness::survives(|dev| {
                    let mut p = vec![bank(0)];
                    for (a, v) in SLOTS.into_iter().zip([1, 3, 7]) {
                        write(&mut p, 24, a, v);
                    }
                    write(&mut p, 24, BASES[0], 5);
                    p.extend(backend::modify_word(BASES[0], sel, Alu::Add, mode, 32, rot).unwrap());
                    run_thread(dev, &p, 1);
                    assert_eq!(
                        debug(dev, BASES[0], 0),
                        config_model(
                            5,
                            [1, 3, 7][if sel == Scratch::CurrentThread {
                                1
                            } else {
                                sel as usize
                            }],
                            Alu::Add,
                            mode,
                            32,
                            rot
                        )
                    );
                });
                assert_eq!(
                    survives,
                    mode == MaskMode::Preserve && rot == 0,
                    "{sel:?} {mode:?} {rot}"
                );
            }
        }
    }
}

#[test]
fn supported_config_add_and_scratch_selectors() {
    harness::in_device(|dev| {
        for t in 0..3 {
            for b in 0..2 {
                for sel in [
                    Scratch::Slot0,
                    Scratch::Slot1,
                    Scratch::Slot2,
                    Scratch::CurrentThread,
                ] {
                    let source = [1, 0x80000001, u32::MAX];
                    let mut p = vec![bank(b)];
                    for (a, v) in SLOTS.into_iter().zip(source) {
                        write(&mut p, 24, a, v);
                    }
                    write(&mut p, 24, BASES[0], u32::MAX);
                    p.extend(
                        backend::modify_word(BASES[0], sel, Alu::Add, MaskMode::Preserve, 32, 0)
                            .unwrap(),
                    );
                    p.extend(backend::read_word(31, BASES[0]).unwrap());
                    p.push(bank(0));
                    p.push(backend::write_word(31, BASES[1]).unwrap());
                    p.push(backend::nop());
                    run_thread(dev, &p, t);
                    let selected = if sel == Scratch::CurrentThread {
                        t
                    } else {
                        sel as usize
                    };
                    assert_eq!(
                        debug(dev, BASES[1], 0),
                        u32::MAX.wrapping_add(source[selected])
                    );
                }
            }
        }
    });
}
#[test]
#[cfg(feature = "silicon")]
fn scratch_mutation_targets_are_global() {
    harness::in_device(|dev| {
        for t in 0..3 {
            for b in 0..2 {
                for (target, addr) in SLOTS.into_iter().enumerate() {
                    let selector = if target == 0 {
                        Scratch::Slot1
                    } else {
                        Scratch::Slot0
                    };
                    for op in ALUS {
                        for mode in [MaskMode::Replace, MaskMode::Preserve] {
                            for width in [1, 32] {
                                for rotation in [0, 31] {
                                    let mut p = vec![bank(b)];
                                    for a in SLOTS {
                                        write(&mut p, 24, a, 0x80000003);
                                    }
                                    write(&mut p, 24, addr, 0xfffffffe);
                                    p.extend(
                                        backend::modify_word(
                                            addr, selector, op, mode, width, rotation,
                                        )
                                        .unwrap(),
                                    );
                                    run_thread(dev, &p, t);
                                    let want = config_model(
                                        0xfffffffe, 0x80000003, op, mode, width, rotation,
                                    );
                                    assert_eq!(debug(dev, addr, b), want);
                                    assert_eq!(debug(dev, addr, 1 - b), want);
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}

#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_debug_read_refusals_have_a_control() {
    assert!(harness::survives(|dev| {
        let mut p = vec![bank(0)];
        write(&mut p, 24, BASES[0], 123);
        run_thread(dev, &p, 1);
        assert_eq!(debug(dev, BASES[0], 0), 123);
    }));
    for addr in SLOTS {
        assert!(!harness::survives(|dev| {
            let mut p = vec![bank(0)];
            write(&mut p, 24, addr, 123);
            run_thread(dev, &p, 1);
            debug(dev, addr, 0);
        }));
    }
    assert!(!harness::survives(|dev| {
        let mut p = vec![bank(1)];
        write(&mut p, 24, BASES[0], 123);
        run_thread(dev, &p, 1);
        debug(dev, BASES[0], 1);
    }));
}
