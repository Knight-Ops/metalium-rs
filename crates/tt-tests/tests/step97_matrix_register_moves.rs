//! Ownership-safe matrix register moves and resident two-stage matrix chains.
use tt_kernels::{
    matmul::Fidelity,
    session::{Session, TileChoice},
};
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
type Transport<'a> = tt_ttsim::LibTtsim<'a>;
#[cfg(feature = "silicon")]
type Transport<'a> = tt_kmd::Kmd;
fn with_session(tiles: usize, f: impl FnOnce(&mut Session<Transport<'_>>)) {
    if let Err(e) = fork_scope(|| {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(tiles),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(tiles),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[test]
fn resident_chains_all_operations_precisions_fidelities() {
    use tt_kernels::matrix_eltwise::{
        MatrixChainTail as Tail, MatrixEltwiseOp as Op, SrcPrecision,
    };

    for tiles in [1, 2] {
        with_session(tiles, |s| {
            for (rows, cols) in [(32, 32), (37, 65)] {
                let a: Vec<_> = (0..rows * cols)
                    .map(|i| 1.0 + (i % 13) as f32 / 8.0)
                    .collect();
                let b: Vec<_> = (0..rows * cols)
                    .map(|i| 0.5 + (i % 7) as f32 / 8.0)
                    .collect();
                let ta = s.upload(&a, rows, cols).unwrap();
                let tb = s.upload(&b, rows, cols).unwrap();
                for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
                    for fidelity in [
                        Fidelity::Lo,
                        Fidelity::HiFi2,
                        Fidelity::HiFi3,
                        Fidelity::HiFi4,
                    ] {
                        for first in [Op::Add, Op::Sub, Op::Mul] {
                            for tail in [
                                Tail::WithRhs(Op::Add),
                                Tail::WithRhs(Op::Sub),
                                Tail::WithRhs(Op::Mul),
                                Tail::WithLhs(Op::Add),
                                Tail::WithLhs(Op::Sub),
                                Tail::WithLhs(Op::Mul),
                                Tail::Square,
                            ] {
                                let transfers = s.dataflow_stats().transfer_packets;
                                let out = s
                                    .matrix_eltwise_chain(
                                        first, tail, &ta, &tb, precision, fidelity,
                                    )
                                    .unwrap();
                                assert_eq!(
                                    s.dataflow_stats().transfer_packets,
                                    transfers,
                                    "chain stages must remain resident"
                                );
                                let got = s.download(&out).unwrap();
                                for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                                    let t = elw_reference(
                                        first,
                                        a,
                                        b,
                                        precision.code(),
                                        fidelity.phases(),
                                    );
                                    let q = f32::from_bits(
                                        t.to_bits()
                                            & if precision == SrcPrecision::Tf32 {
                                                0xffff_e000
                                            } else {
                                                0xffff_0000
                                            },
                                    );
                                    let (op, lhs, rhs) = match tail {
                                        Tail::WithRhs(op) => (op, q, b),
                                        Tail::WithLhs(op) => (op, a, q),
                                        Tail::Square => (Op::Mul, q, q),
                                    };
                                    let want = elw_reference(
                                        op,
                                        lhs,
                                        rhs,
                                        precision.code(),
                                        fidelity.phases(),
                                    );
                                    assert_eq!(got[i].to_bits(), want.to_bits(), "{tiles} tiles {rows}x{cols} {precision:?} {fidelity:?} {first:?} {tail:?} at {i}");
                                }
                                s.free(out).unwrap();
                            }
                        }
                    }
                }
                s.free(ta).unwrap();
                s.free(tb).unwrap();
            }
        });
    }
}

// Normal finite dyadic domain: independent f64 grid arithmetic, with each
// phase rounded at the F32 accumulation boundary. The SrcA low TF32 bit is
// unused, as measured by step90. Add/sub share a 10-bit alignment quantum.
fn elw_reference(
    op: tt_kernels::matrix_eltwise::MatrixEltwiseOp,
    a: f32,
    b: f32,
    precision: u32,
    phases: u32,
) -> f32 {
    use tt_kernels::matrix_eltwise::MatrixEltwiseOp as Op;
    fn truncate(x: f64, bits: i32) -> f64 {
        if x == 0.0 {
            return x;
        }
        let quantum = 2f64.powi(x.abs().log2().floor() as i32 - bits);
        (x / quantum).trunc() * quantum
    }
    let bits = if precision == 4 { 10 } else { 7 };
    let a = truncate(f64::from(a), bits);
    let b = truncate(f64::from(b), bits);
    if op == Op::Mul {
        let a = truncate(a, if bits == 10 { 9 } else { 7 });
        let ah = truncate(a, 4);
        let bh = truncate(b, 6);
        let products = [ah * bh, (a - ah) * bh, ah * (b - bh), (a - ah) * (b - bh)];
        products[..phases as usize]
            .iter()
            .fold(0f32, |acc, &p| (f64::from(acc) + p) as f32)
    } else {
        let magnitude = a.abs().max(b.abs());
        if magnitude == 0.0 {
            return 0.0;
        }
        let quantum = 2f64.powi(magnitude.log2().floor() as i32 - 10);
        let a = (a / quantum).trunc() * quantum;
        let b = (b / quantum).trunc() * quantum;
        (if op == Op::Add { a + b } else { a - b }) as f32
    }
}

/// Supported four-row moves use a loaded bank; SFPU writes preserve all F32
/// bits until the move. One-row variants are silicon-only (divergence row 37).
fn instruction_roundtrip(four: bool, address_case: u32) {
    use tt_isa::{
        backend::{self, Before, ConfigWords},
        cfg::generated::{alu, thread},
        isa::generated::encode,
        matrix::Banks,
        sfpu,
    };
    use tt_kernels::{datapath as d, matmul};
    use tt_tests::harness::{self, Roles, Run};
    let values = [
        0x3f800000, 0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x807fffff, 0x00800000,
        0x3f801fff, 0x3f802000, 0x3f80ffff, 0x3f810000, 0x7f800000, 0xff800000, 0x7fc12345,
        0xffc76543, 0x7f7fffff,
    ];
    harness::in_device(|dev| {
        for precision in [4, 5] {
            for which in 0..3 {
                for mask in if cfg!(feature = "silicon") {
                    vec![0u32, 1, 2, 3]
                } else {
                    vec![0]
                } {
                    for flush in [false, true] {
                        for x in if address_case == 0 && mask == 0 {
                            values.to_vec()
                        } else {
                            vec![0x3f800000]
                        } {
                            let dst_edge = address_case & 1 != 0;
                            let src_edge = address_case & 2 != 0;
                            let src_arg = if src_edge { 63 } else { 4 };
                            let dst_arg = if dst_edge { 3 } else { 0 };

                            let staged = matmul::stage_operand(0, &[[0f32; 16]; 16]).0;
                            let mut up = matmul::unpack_prelude(matmul::Operands {
                                a_addr: d::STAGE,
                                na: 256,
                                b_addr: d::STAGE + 0x2000,
                                nb: 256,
                                out: precision,
                            });
                            // ADCs survive programs on silicon. Reinitialize all
                            // dimensions before each flat source-bank refill.
                            up.push(
                                encode::Setadcxy::ZERO
                                    .u0(1)
                                    .u1(1)
                                    .x0(1)
                                    .x1(1)
                                    .y0(1)
                                    .y1(1)
                                    .encode()
                                    .unwrap(),
                            );
                            up.push(
                                encode::Setadczw::ZERO
                                    .u0(1)
                                    .u1(1)
                                    .z0(1)
                                    .z1(1)
                                    .w0(1)
                                    .w1(1)
                                    .encode()
                                    .unwrap(),
                            );
                            let base = encode::UnpacrRegular::ZERO.multi_context_mode(1);
                            up.push(d::set_adc_x(d::Unpacker::SrcA, 0, 255));
                            up.push(d::set_adc_x(d::Unpacker::SrcB, 0, 255));
                            let (i, banks) = Banks::after_reset().unpack_a(base).unwrap();
                            up.push(i);
                            let (i, mut banks) = banks.unpack_b(base).unwrap();
                            up.push(i);
                            up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                            up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
                            let mut math = matmul::math_prelude();
                            math.push(d::thread_entry(thread::DISABLE_IMPLIED_SRCA_FMT_Base, 1));
                            math.push(d::thread_entry(thread::DISABLE_IMPLIED_SRCB_FMT_Base, 1));
                            let mut config = ConfigWords::new();
                            config
                                .set(alu::ALU_FORMAT_SPEC_REG_SrcA_override, 1)
                                .unwrap();
                            config
                                .set(alu::ALU_FORMAT_SPEC_REG_SrcA_val, precision)
                                .unwrap();
                            config
                                .set(alu::ALU_FORMAT_SPEC_REG_SrcB_override, 1)
                                .unwrap();
                            config
                                .set(alu::ALU_FORMAT_SPEC_REG_SrcB_val, precision)
                                .unwrap();
                            // Test MOVB2A's zero-exponent flush separately from
                            // the readback move, which preserves subnormal bits.
                            config.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
                            config
                                .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, u32::from(!flush))
                                .unwrap();
                            #[cfg(feature = "silicon")]
                            config.set(alu::DEST_REGW_BASE_Base, 0).unwrap();
                            math.extend(d::config_program(&config));
                            // The simulator has no implementation of this
                            // explicit predicate-disable form (step97 addendum).
                            #[cfg(feature = "silicon")]
                            math.push(encode::sfpencc(0, 0, 2).unwrap());
                            math.push(encode::sfpconfig(0, 15, 1).unwrap());
                            math.extend(sfpu::load_f32(1, 0).unwrap());
                            for cols in [0, sfpu::DST_ODD_COLUMNS] {
                                math.push(
                                    sfpu::store(1, sfpu::store_format::INT32, 0, cols).unwrap(),
                                );
                            }
                            math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
                            for row in [0, 4, 8, 12] {
                                let (i, next) = banks
                                    .movd2a(encode::Movd2A::ZERO.move4_rows(1).src_row(row))
                                    .unwrap();
                                math.push(i);
                                banks = next;
                                let (i, next) = banks
                                    .movd2b(encode::Movd2B::ZERO.move4_rows(1).src_row(row))
                                    .unwrap();
                                math.push(i);
                                banks = next;
                            }
                            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                            let (i, next) = banks
                                .movb2d(encode::Movb2D::ZERO.move4_rows(1).src_row(4).dst_row(12))
                                .unwrap();
                            math.push(i);
                            banks = next;
                            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                            math.extend(sfpu::load_f32(0, x).unwrap());
                            for cols in [0, sfpu::DST_ODD_COLUMNS] {
                                math.push(
                                    sfpu::store(0, sfpu::store_format::INT32, 0, 8 | cols).unwrap(),
                                );
                            }
                            math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
                            // Effective Dst row 11 aligns to 8; effective Src row
                            // 71 wraps to 7. Silicon does not align a write-side
                            // Src address; the read side of B2A aligns to 4.
                            math.push(d::thread_entry(
                                thread::DEST_TARGET_REG_CFG_MATH_Offset,
                                if dst_edge { 4 } else { 0 },
                            ));
                            math.push(
                                encode::Setrwc::ZERO
                                    .dst(1)
                                    .dst_val(if dst_edge { 4 } else { 8 })
                                    .src_a(1)
                                    .src_a_val(if src_edge { 8 } else { 0 })
                                    .src_b(1)
                                    .src_b_val(if src_edge { 8 } else { 0 })
                                    .encode()
                                    .unwrap(),
                            );
                            math.push(encode::sfpconfig(mask << 9, 15, 1).unwrap());
                            math.extend([encode::sfpnop().unwrap(); 4]);
                            math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
                            let (i, next) = if which == 0 {
                                banks.movd2a(
                                    encode::Movd2A::ZERO
                                        .move4_rows(u32::from(four))
                                        .src_row(src_arg)
                                        .dst_row(dst_arg),
                                )
                            } else {
                                banks.movd2b(
                                    encode::Movd2B::ZERO
                                        .move4_rows(u32::from(four))
                                        .src_row(src_arg)
                                        .dst_row(dst_arg),
                                )
                            }
                            .unwrap();
                            math.push(i);
                            banks = next;
                            if which == 2 {
                                let (i, next) = banks.movb2a(src_arg, 0, four, src_arg).unwrap();
                                math.push(i);
                                banks = next;
                            }
                            // The Matrix and SFPU queues run independently. Keep
                            // masks and formats stable until the move completes.
                            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                            let mut readback_config = ConfigWords::new();
                            readback_config
                                .set(alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                                .unwrap();
                            readback_config
                                .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 1)
                                .unwrap();
                            math.extend(d::config_program(&readback_config));
                            math.push(encode::sfpconfig(0, 15, 1).unwrap());
                            math.extend([encode::sfpnop().unwrap(); 4]);
                            math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
                            math.push(d::thread_entry(thread::DEST_TARGET_REG_CFG_MATH_Offset, 0));
                            math.push(
                                encode::Setrwc::ZERO
                                    .dst(1)
                                    .dst_val(0)
                                    .src_a(1)
                                    .src_a_val(0)
                                    .src_b(1)
                                    .src_b_val(0)
                                    .encode()
                                    .unwrap(),
                            );
                            // ZEROACC invalidates rows; it does not scrub the
                            // underlying bits read by the harness's SFPU dump.
                            math.extend(sfpu::load_f32(1, 0).unwrap());
                            for row in [0, 4, 8].into_iter().take(if src_edge { 3 } else { 2 }) {
                                for cols in [0, sfpu::DST_ODD_COLUMNS] {
                                    math.push(
                                        sfpu::store(1, sfpu::store_format::INT32, 0, row | cols)
                                            .unwrap(),
                                    );
                                }
                            }
                            math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
                            if which == 1 {
                                for row in [0, 4, 8].into_iter().take(if src_edge { 3 } else { 2 })
                                {
                                    let (i, next) = banks
                                        .movb2d(
                                            encode::Movb2D::ZERO
                                                .move4_rows(1)
                                                .src_row(row)
                                                .dst_row(row),
                                        )
                                        .unwrap();
                                    math.push(i);
                                    banks = next;
                                }
                            } else {
                                let (i, next) =
                                    banks.mova2d(encode::Mova2D::ZERO.move8_rows(1)).unwrap();
                                math.push(i);
                                banks = next;
                                if src_edge {
                                    for row in 8..12 {
                                        let (i, next) = banks
                                            .mova2d(encode::Mova2D::ZERO.src_row(row).dst_row(row))
                                            .unwrap();
                                        math.push(i);
                                        banks = next;
                                    }
                                }
                            }
                            let (i, next) = banks.release_a().unwrap();
                            math.push(i);
                            let (i, _) = next.release_b().unwrap();
                            math.push(i);
                            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                            let result = harness::run(
                                dev,
                                &Run::roles(Roles {
                                    unpack: &up,
                                    math: &math,
                                    pack: &[],
                                })
                                .stage(&[(d::STAGE, &staged), (d::STAGE + 0x2000, &staged)])
                                .dump_rows(16),
                            );
                            for row in 12..16 {
                                for col in 0..16 {
                                    assert_eq!(result.dst_at(row, col), 0, "Src refill: address_case={address_case} move={which} mask={mask} row={row} col={col}");
                                }
                            }
                            for row in if src_edge { 0..0 } else { 8..12 } {
                                assert_eq!(
                                    result.dst_at(row, 0),
                                    x,
                                    "Dst setup row={row} x={x:08x}"
                                );
                            }
                            let x = if which == 2 && flush && x & 0x7f800000 == 0 {
                                0
                            } else {
                                x
                            };
                            let q = x & if precision == 4 {
                                0xffffe000
                            } else {
                                0xffff0000
                            };
                            for row in 0..if src_edge { 12 } else { 8 } {
                                for col in 0..16 {
                                    let written = if four {
                                        let start = if src_edge { 7 } else { 4 };
                                        row >= start && row < start + 4
                                    } else {
                                        row == if src_edge { 7 } else { 4 }
                                    };
                                    // Blackhole aligns the read side only. B2A
                                    // reads B rows 4..7 into unaligned A rows 7..10.
                                    let written =
                                        written && !(src_edge && four && which == 2 && row != 10);
                                    let want = if written && mask & (1 << (col & 1)) == 0 {
                                        q
                                    } else {
                                        0
                                    };
                                    assert_eq!(result.dst_at(row, col), want,
                                    "address_case={address_case} move {which} four={four} precision={precision} mask={mask} x={x:08x} row={row} col={col}");
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
fn four_row_conversion_alignment_wrapping_and_lane_masks() {
    instruction_roundtrip(true, 0);
}

#[test]
#[cfg(feature = "silicon")]
fn one_row_conversion_alignment_wrapping_and_lane_masks() {
    for case in 0..4 {
        instruction_roundtrip(false, case);
    }
}

#[test]
#[cfg(feature = "silicon")]
fn four_row_rwc_alignment_and_wrapping() {
    for case in 1..4 {
        instruction_roundtrip(true, case);
    }
}

#[test]
fn changed_input_traces_hold_freed_operands_and_preserve_padding() {
    use tt_kernels::{
        matrix_eltwise::{MatrixChainTail as Tail, MatrixEltwiseOp as Op, SrcPrecision},
        tensor::Pad,
    };
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            let dims = [37, 65];
            let a = s
                .upload(&vec![2.0; dims[0] * dims[1]], dims[0], dims[1])
                .unwrap();
            let b = s
                .upload(&vec![3.0; dims[0] * dims[1]], dims[0], dims[1])
                .unwrap();
            let before = (a.pad(), b.pad());
            s.begin_trace().unwrap();
            let product = s
                .matrix_eltwise_chain(
                    Op::Mul,
                    Tail::WithRhs(Op::Sub),
                    &a,
                    &b,
                    SrcPrecision::Tf32,
                    Fidelity::HiFi4,
                )
                .unwrap();
            assert_eq!(product.pad(), Pad::Undefined);
            let sum = s.sum_rows(&product).unwrap();
            s.free(product).unwrap();
            let trace = s.end_trace().unwrap();
            assert_eq!((a.pad(), b.pad()), before);
            s.free(b).unwrap();
            for value in [2.0, 5.0, -3.0] {
                s.write(&a, &vec![value; dims[0] * dims[1]]).unwrap();
                s.replay(trace).unwrap();
                assert_eq!(
                    s.download(&sum).unwrap(),
                    vec![(value - 1.0) * 3.0 * dims[0] as f32; dims[1]]
                );
                assert_eq!(a.pad(), before.0);
            }
            s.release_trace(trace).unwrap();
            s.free(sum).unwrap();
            s.free(a).unwrap();
        });
    }
}

#[test]
fn poisoned_padding_downstream_matmul_and_reduction() {
    use tt_kernels::{
        matrix_eltwise::{MatrixChainTail as Tail, MatrixEltwiseOp as Op, SrcPrecision},
        tensor::Pad,
    };
    with_session(2, |s| {
        let (rows, cols) = (37usize, 65usize);
        let a = s.upload(&vec![2.0; rows * cols], rows, cols).unwrap();
        let b = s.upload(&vec![3.0; rows * cols], rows, cols).unwrap();
        a.set_pad(Pad::Undefined);
        b.set_pad(Pad::Undefined);
        s.sync().unwrap();
        let w = s
            .device()
            .alloc_window(tt_device::tlb::WindowKind::FourGib)
            .unwrap();
        let mut snapshots = vec![];
        for t in [&a, &b] {
            for tile in 0..t.placement.tiles() {
                let slot = t.placement.slot(tile);
                let mut bytes = vec![0; slot.len() as usize];
                s.device().dram_read(&w, slot, &mut bytes).unwrap();
                for r in 0..32 {
                    for c in 0..32 {
                        if tile / cols.div_ceil(32) * 32 + r >= rows
                            || tile % cols.div_ceil(32) * 32 + c >= cols
                        {
                            let at = 16 + tt_isa::dm::face_index(r, c) * 4;
                            bytes[at..at + 4].copy_from_slice(&0x7fc12345u32.to_le_bytes());
                        }
                    }
                }
                s.device().dram_write(&w, slot, &bytes).unwrap();
                snapshots.push((slot, bytes));
            }
        }
        for tail in [Tail::WithRhs(Op::Mul), Tail::WithLhs(Op::Mul), Tail::Square] {
            let out = s
                .matrix_eltwise_chain(Op::Add, tail, &a, &b, SrcPrecision::Tf32, Fidelity::HiFi4)
                .unwrap();
            assert_eq!(out.pad(), Pad::Undefined);
            let value = match tail {
                Tail::WithRhs(_) => 15.0,
                Tail::WithLhs(_) => 10.0,
                Tail::Square => 25.0,
            };
            let sum = s.sum_rows(&out).unwrap();
            assert_eq!(s.download(&sum).unwrap(), vec![value * rows as f32; cols]);
            // A fresh undefined-padding output makes matmul establish its own
            // masking contract, independently of the preceding reduction.
            let matmul_input = s
                .matrix_eltwise_chain(Op::Add, tail, &a, &b, SrcPrecision::Tf32, Fidelity::HiFi4)
                .unwrap();
            assert_eq!(matmul_input.pad(), Pad::Undefined);
            let weights = s.upload(&vec![1.0; cols * 3], cols, 3).unwrap();
            let product = s
                .matmul_dram(
                    &matmul_input,
                    false,
                    &weights,
                    false,
                    tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    20_000_000,
                )
                .unwrap();
            assert_eq!(
                s.download(&product).unwrap(),
                vec![value * cols as f32; rows * 3]
            );
            s.free(product).unwrap();
            s.free(matmul_input).unwrap();
            s.free(weights).unwrap();
            s.free(sum).unwrap();
            s.free(out).unwrap();
        }
        assert_eq!((a.pad(), b.pad()), (Pad::Undefined, Pad::Undefined));
        for (slot, want) in snapshots {
            let mut bytes = vec![0; slot.len() as usize];
            s.device().dram_read(&w, slot, &mut bytes).unwrap();
            assert_eq!(bytes, want);
        }
        let bad = s.upload(&vec![1.0; cols], 1, cols).unwrap();
        assert!(s
            .matrix_eltwise_chain(
                Op::Add,
                Tail::Square,
                &a,
                &bad,
                SrcPrecision::Tf32,
                Fidelity::Lo
            )
            .is_err());
        s.free(bad).unwrap();
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}

#[test]
fn safe_negative_controls_detect_register_reuse_errors() {
    use tt_isa::{
        backend::{self, Before, ConfigWords},
        cfg::generated::{alu, thread},
        isa::generated::encode,
        matrix::{Banks, SrcBroadcast},
    };
    use tt_kernels::{datapath as d, matmul};
    use tt_tests::harness::{self, Roles, Run};
    // Every row/column differs. All intermediates and phase-zero products are
    // exactly representable, so these comparisons require no error budget.
    for mutation in 0..5 {
        harness::in_device(|dev| {
            let a = core::array::from_fn::<_, 16, _>(|r| {
                core::array::from_fn::<_, 16, _>(|c| (1 + r + c % 3) as f32)
            });
            let b = [[2f32; 16]; 16];
            let sa = matmul::stage_operand(0, &a).0;
            let sb = matmul::stage_operand(0, &b).0;
            let mut up = matmul::unpack_prelude(matmul::Operands {
                a_addr: d::STAGE,
                na: 256,
                b_addr: d::STAGE + 0x2000,
                nb: 256,
                out: 4,
            });
            let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
            up.push(d::set_adc_x(d::Unpacker::SrcA, 0, 255));
            up.push(d::set_adc_x(d::Unpacker::SrcB, 0, 255));
            let (i, banks) = Banks::after_reset().unpack_a(unpack).unwrap();
            up.push(i);
            let (i, mut banks) = banks.unpack_b(unpack).unwrap();
            up.push(i);
            up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
            up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
            let mut math = matmul::math_prelude();
            math.push(d::thread_entry(thread::DISABLE_IMPLIED_SRCA_FMT_Base, 1));
            math.push(d::thread_entry(thread::DISABLE_IMPLIED_SRCB_FMT_Base, 1));
            let mut config = ConfigWords::new();
            config
                .set(alu::ALU_FORMAT_SPEC_REG_SrcA_override, 1)
                .unwrap();
            config.set(alu::ALU_FORMAT_SPEC_REG_SrcA_val, 4).unwrap();
            config
                .set(alu::ALU_FORMAT_SPEC_REG_SrcB_override, 1)
                .unwrap();
            config.set(alu::ALU_FORMAT_SPEC_REG_SrcB_val, 4).unwrap();
            config.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
            #[cfg(feature = "silicon")]
            config.set(alu::DEST_REGW_BASE_Base, 0).unwrap();
            math.extend(d::config_program(&config));
            let (i, next) = banks
                .elwadd(encode::Elwadd::ZERO, SrcBroadcast::None)
                .unwrap();
            math.push(i);
            banks = next;
            if mutation != 1 {
                // omitted move
                for row in [0, 4] {
                    let (i, next) = if mutation == 2 {
                        // swapped destination
                        banks.movd2b(encode::Movd2B::ZERO.move4_rows(1).src_row(row).dst_row(row))
                    } else {
                        banks.movd2a(
                            encode::Movd2A::ZERO
                                .move4_rows(1)
                                .src_row(row + if mutation == 3 { 8 } else { 0 })
                                .dst_row(row),
                        )
                    }
                    .unwrap();
                    math.push(i);
                    banks = next;
                }
            }
            if mutation != 4 {
                // stale stage-two accumulator
                for row in 0..8 {
                    math.push(encode::zeroacc(0, 0, 0, row).unwrap());
                }
            }
            math.push(encode::Setrwc::ZERO.fidelity(1).encode().unwrap());
            let (i, _) = banks
                .elwmul_release_both(encode::Elwmul::ZERO, SrcBroadcast::None)
                .unwrap();
            math.push(i);
            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
            let result = harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &up,
                    math: &math,
                    pack: &[],
                })
                .stage(&[(d::STAGE, &sa), (d::STAGE + 0x2000, &sb)])
                .dump_rows(8),
            );
            let matches = (0..8)
                .all(|r| (0..16).all(|c| result.dst_at(r, c) == ((a[r][c] + 2.0) * 2.0).to_bits()));
            if mutation == 0 {
                for (r, row) in a.iter().enumerate().take(8) {
                    for (c, &value) in row.iter().enumerate() {
                        assert_eq!(
                            result.dst_at(r, c),
                            ((value + 2.0) * 2.0).to_bits(),
                            "control r={r} c={c}"
                        );
                    }
                }
            }
            assert_eq!(matches, mutation == 0, "negative control {mutation}");
        });
    }
}

#[test]
fn parent_view_and_alternating_precision_keep_state_and_padding() {
    use tt_kernels::matrix_eltwise::{
        MatrixChainTail as Tail, MatrixEltwiseOp as Op, SrcPrecision,
    };
    with_session(2, |s| {
        let parent = s.upload(&vec![2.0; 69 * 65], 69, 65).unwrap();
        let view = parent.rows_view(32, 37).unwrap();
        let b = s.upload(&vec![3.0; 37 * 65], 37, 65).unwrap();
        let before = parent.pad();
        for precision in [SrcPrecision::Bf16, SrcPrecision::Tf32] {
            let out = s
                .matrix_eltwise_chain(Op::Add, Tail::Square, &view, &b, precision, Fidelity::HiFi4)
                .unwrap();
            let sum = s.sum_rows(&out).unwrap();
            assert_eq!(s.download(&sum).unwrap(), vec![25.0 * 37.0; 65]);
            // Ordinary matrix consumers after a chain establish fresh formats.
            let regular = s
                .matrix_eltwise(Op::Mul, &view, &b, SrcPrecision::Tf32, Fidelity::HiFi4)
                .unwrap();
            assert_eq!(s.download(&regular).unwrap(), vec![6.0; 37 * 65]);
            assert_eq!(parent.pad(), before);
            assert_eq!(s.download(&parent).unwrap(), vec![2.0; 69 * 65]);
            s.free(regular).unwrap();
            s.free(sum).unwrap();
            s.free(out).unwrap();
        }
        s.free(view).unwrap();
        s.free(parent).unwrap();
        s.free(b).unwrap();
    });
}

#[test]
fn pinned_move_models_preserve_raw_formats_and_masks() {
    use tt_isa::matrix::moves::{self, Style};
    assert_eq!(moves::rows(1027, 71, true), (0, 7, 4));
    assert_eq!(moves::rows(1027, 71, false), (3, 7, 1));
    assert_eq!(
        moves::dst_to_src(0x807f, false, false, Style::Bf16),
        Some(0x4007f)
    );
    assert_eq!(
        moves::dst_to_src(0xffe1, false, false, Style::Fp16),
        Some(0x7ff01)
    );
    assert_eq!(moves::dst_to_src(0, false, true, Style::Bf16), None);
    assert_eq!(moves::dst_to_src(0, false, false, Style::Tf32), None);
    // TF32's three low fraction bits move above its eight exponent bits.
    assert_eq!(
        moves::dst_to_src(0x00ff_e000, true, false, Style::Tf32),
        Some(0x7ff)
    );
    assert_eq!(
        moves::dst_to_src(0xffff_1abc, true, true, Style::Tf32),
        Some(0x1abc)
    );
    for x in [0, 0x40000, 0x3ff00, 0x7ff00] {
        assert_eq!(moves::b_to_a(x, true), 0);
        assert_eq!(moves::b_to_a(x, false), x);
    }
    for x in [0x7ffff, 0x40001, 0x12345] {
        assert_eq!(moves::b_to_a(x, true), x);
    }
    let b = [[0x40000; 16]; 64];
    let mut a = [[0x123; 16]; 64];
    moves::copy_b(&b, &mut a, 71, 127, true, [2; 8], true);
    for (r, row) in a.iter().enumerate() {
        for (c, &x) in row.iter().enumerate() {
            assert_eq!(
                x,
                if (7..11).contains(&r) && c % 2 == 0 {
                    0
                } else {
                    0x123
                }
            );
        }
    }
    let dst = [[0xffff_ffff; 16]; 1024];
    let mut src = [[0x123; 16]; 64];
    moves::copy_dst(
        &dst,
        &mut src,
        (1027, 71, true),
        [1; 8],
        Style::Bf16,
        true,
        false,
    )
    .unwrap();
    for (r, row) in src.iter().enumerate() {
        for (c, &x) in row.iter().enumerate() {
            assert_eq!(
                x,
                if (7..11).contains(&r) && c % 2 == 1 {
                    0x7f8ff
                } else {
                    0x123
                }
            );
        }
    }
}

#[test]
#[ignore = "benchmark"]
#[cfg(feature = "silicon")]
fn chain_vs_separate_matrix_calls_release_baseline() {
    use std::time::Instant;
    use tt_kernels::matrix_eltwise::{
        MatrixChainTail as Tail, MatrixEltwiseOp as Op, SrcPrecision,
    };
    let release = !cfg!(debug_assertions);
    assert!(release, "benchmarks require --release");
    // End libtest's test-name line so every BENCH record has its own line.
    println!("\nMEASURE resident matrix chains versus separate calls");
    with_session(2, |s| {
        for (rows, cols) in [(64, 64), (65, 70)] {
            let a = s.upload(&vec![2.0; rows * cols], rows, cols).unwrap();
            let b = s.upload(&vec![3.0; rows * cols], rows, cols).unwrap();
            for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
                for chain in [false, true] {
                    let before = s.dataflow_stats().clone();
                    let mut samples = vec![];
                    for rep in 0..9 {
                        let start = Instant::now();
                        let mut temporary = None;
                        let out = if chain {
                            s.matrix_eltwise_chain(
                                Op::Add,
                                Tail::WithRhs(Op::Mul),
                                &a,
                                &b,
                                precision,
                                Fidelity::HiFi4,
                            )
                            .unwrap()
                        } else {
                            let t = s
                                .matrix_eltwise(Op::Add, &a, &b, precision, Fidelity::HiFi4)
                                .unwrap();
                            let out = s
                                .matrix_eltwise(Op::Mul, &t, &b, precision, Fidelity::HiFi4)
                                .unwrap();
                            temporary = Some(t);
                            out
                        };
                        s.sync().unwrap();
                        let elapsed = start.elapsed().as_secs_f64();
                        assert_eq!(s.download(&out).unwrap(), vec![15.0; rows * cols]);
                        s.free(out).unwrap();
                        if let Some(t) = temporary {
                            s.free(t).unwrap();
                        }
                        if rep >= 2 {
                            samples.push(elapsed);
                        }
                    }
                    samples.sort_by(f64::total_cmp);
                    println!("BENCH {{\"kind\":\"matrix_chain\",\"git\":\"{}\",\"card\":{},\"rows\":{rows},\"cols\":{cols},\"chain\":{chain},\"tiles\":2,\"precision\":\"{precision:?}\",\"fidelity\":\"HiFi4\",\"warmups\":2,\"samples\":7,\"key\":\"matrix_chain_{rows}x{cols}_{precision:?}_{chain}\",\"timed\":\"host dispatch through sync\",\"median\":{},\"unit\":\"us\",\"p10\":{},\"p90\":{},\"regions\":{},\"batches\":{},\"transfers\":{}}}",
                    tt_tests::bench::git_sha(), tt_tests::backend::device_index(), samples[3]*1e6, samples[1]*1e6, samples[5]*1e6,
                    s.dataflow_stats().regions - before.regions, s.dataflow_stats().batches - before.batches,
                    s.dataflow_stats().transfer_packets - before.transfer_packets);
                }
            }
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

#[test]
fn finite_error_bounds_include_both_stages_and_intermediate_conversion() {
    use tt_kernels::matrix_eltwise::{
        MatrixChainTail as Tail, MatrixEltwiseOp as Op, SrcPrecision,
    };
    fn ideal(op: Op, a: f64, b: f64) -> f64 {
        match op {
            Op::Add => a + b,
            Op::Sub => a - b,
            Op::Mul => a * b,
        }
    }
    fn trunc(x: f64, bits: i32) -> f64 {
        if x == 0.0 {
            return x;
        }
        let q = 2f64.powi(x.abs().log2().floor() as i32 - bits);
        (x / q).trunc() * q
    }
    fn bound(op: Op, a: f64, b: f64, bits: i32, phases: usize) -> f64 {
        let (qa, qb) = (trunc(a, bits), trunc(b, bits));
        if op == Op::Mul {
            let qa = trunc(qa, if bits == 10 { 9 } else { 7 });
            let (ah, bh) = (trunc(qa, 4), trunc(qb, 6));
            let products = [
                ah * bh,
                (qa - ah) * bh,
                ah * (qb - bh),
                (qa - ah) * (qb - bh),
            ];
            let kept: f64 = products[..phases].iter().sum();
            // Triangle bound for input conversion, omitted phases and eight
            // rounded operations (four products/four sums) at FP32 precision.
            let u = 2f64.powi(-24);
            let gamma8 = 8.0 * u / (1.0 - 8.0 * u);
            (a * b - qa * qb).abs()
                + (qa * qb - kept).abs()
                + gamma8 * products.iter().map(|p| p.abs()).sum::<f64>()
        } else {
            let mag = qa.abs().max(qb.abs());
            let quantum = if mag == 0.0 {
                0.0
            } else {
                2f64.powi(mag.log2().floor() as i32 - 10)
            };
            (a - qa).abs() + (b - qb).abs() + 2.0 * quantum
        }
    }
    fn worst_stage(op: Op, a_max: f64, b_max: f64, bits: i32, phases: u32) -> f64 {
        let u = 2f64.powi(-bits);
        if op == Op::Mul {
            // Uniform bound over the whole possible intermediate interval:
            // fidelity truncation is discontinuous at a fraction grid boundary.
            // The unconsumed TF32 SrcA bit costs at most 2^-9 relatively.
            let a_loss = u + if bits == 10 { 2f64.powi(-9) } else { 0.0 };
            let omitted = match phases {
                1 => 2f64.powi(-4) + 2f64.powi(-6),
                2 => 2f64.powi(-6),
                3 => 2f64.powi(-10),
                _ => 0.0,
            };
            let round = 8.0 * 2f64.powi(-24) / (1.0 - 8.0 * 2f64.powi(-24));
            a_max * b_max * (a_loss + u + omitted + round)
        } else {
            let mag = a_max.max(b_max);
            let q = if mag == 0.0 {
                0.0
            } else {
                2f64.powi(mag.log2().floor() as i32 - 10)
            };
            u * (a_max + b_max) + 2.0 * q
        }
    }
    with_session(2, |s| {
        let a: Vec<_> = (0usize..37 * 65)
            .map(|i| (1.0 + (i % 31) as f32 / 32.0) * 2f32.powi((i % 17) as i32 - 8))
            .collect();
        let b: Vec<_> = (0usize..37 * 65)
            .map(|i| (1.0 + (i % 29) as f32 / 32.0) * 2f32.powi((i * 7 % 17) as i32 - 8))
            .collect();
        let ta = s.upload(&a, 37, 65).unwrap();
        let tb = s.upload(&b, 37, 65).unwrap();
        for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
            let bits = if precision == SrcPrecision::Tf32 {
                10
            } else {
                7
            };
            for fidelity in [
                Fidelity::Lo,
                Fidelity::HiFi2,
                Fidelity::HiFi3,
                Fidelity::HiFi4,
            ] {
                for first in [Op::Add, Op::Sub, Op::Mul] {
                    for tail in [
                        Tail::WithRhs(Op::Add),
                        Tail::WithRhs(Op::Sub),
                        Tail::WithRhs(Op::Mul),
                        Tail::WithLhs(Op::Add),
                        Tail::WithLhs(Op::Sub),
                        Tail::WithLhs(Op::Mul),
                        Tail::Square,
                    ] {
                        let out = s
                            .matrix_eltwise_chain(first, tail, &ta, &tb, precision, fidelity)
                            .unwrap();
                        let got = s.download(&out).unwrap();
                        for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                            let (a, b) = (f64::from(a), f64::from(b));
                            let exact_t = ideal(first, a, b);
                            let first_error = bound(first, a, b, bits, fidelity.phases() as usize);
                            let t_max = exact_t.abs() + first_error;
                            // Q truncates toward zero: no magnitude growth, and
                            // at most one fraction quantum at the largest exponent.
                            let conversion = if t_max == 0.0 {
                                0.0
                            } else {
                                2f64.powi(t_max.log2().floor() as i32 - bits)
                            };
                            let e = first_error + conversion;
                            let (exact, propagated, stage2) = match tail {
                                Tail::WithRhs(op) => (
                                    ideal(op, exact_t, b),
                                    e * if op == Op::Mul { b.abs() } else { 1.0 },
                                    worst_stage(op, t_max, b.abs(), bits, fidelity.phases()),
                                ),
                                Tail::WithLhs(op) => (
                                    ideal(op, a, exact_t),
                                    e * if op == Op::Mul { a.abs() } else { 1.0 },
                                    worst_stage(op, a.abs(), t_max, bits, fidelity.phases()),
                                ),
                                Tail::Square => (
                                    exact_t * exact_t,
                                    e * (2.0 * exact_t.abs() + e),
                                    worst_stage(Op::Mul, t_max, t_max, bits, fidelity.phases()),
                                ),
                            };
                            assert!((f64::from(got[i])-exact).abs() <= propagated+stage2,
                                "{first:?} {tail:?} {precision:?} {fidelity:?} at {i}: {} vs {exact}, bound {}", got[i], propagated+stage2);
                        }
                        s.free(out).unwrap();
                    }
                }
            }
        }
        s.free(ta).unwrap();
        s.free(tb).unwrap();
    });
}
