//! Device sort kernel: stable, deterministic, against an independent host
//! oracle. Every problem is one lane of a plane; the oracle is a stable sort on
//! `(key, original index)` that shares nothing with the kernel but the layout
//! function `plane_coord`, which the Burn-level gates (step136) check from the
//! outside.
use tt_kernels::{
    session::{Session, TileChoice},
    sfpu::sort::{plane_coord, plane_dims, reference, Spec},
    tensor::Elem,
};

/// The session of a gate: ttsim by default, a card with `silicon`.
macro_rules! session {
    ($s:ident) => {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut $s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut $s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap();
        $s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
    };
}

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

/// F32 words that stress `total_cmp`: both zeros, subnormals, NaNs of both
/// signs and payloads, infinities, extremes, repeats.
const F32_SPECIALS: [u32; 20] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7f80_0001,
    0xff80_0001,
    0x7fff_ffff,
    0xffff_ffff,
    0x7fc1_2345,
    0xffc5_4321,
];

const I32_SPECIALS: [u32; 12] = [
    0,
    1,
    u32::MAX,
    i32::MIN as u32,
    i32::MAX as u32,
    i32::MIN as u32 + 1,
    i32::MAX as u32 - 1,
    0x0000_ffff,
    0xffff_0000,
    0x7fff_ffff,
    0x8000_0001,
    0x0100_0000,
];

fn problems(elem: Elem, count: usize, n: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut s = seed;
    let specials: &[u32] = if elem == Elem::I32 {
        &I32_SPECIALS
    } else {
        &F32_SPECIALS
    };
    (0..count)
        .map(|q| {
            (0..n)
                .map(|e| match q % 4 {
                    0 => specials[(e * 5 + q) % specials.len()],
                    // Few distinct values: ties throughout.
                    1 => specials[(lcg(&mut s) % 3) as usize],
                    2 => lcg(&mut s),
                    _ => specials[lcg(&mut s) as usize % specials.len()],
                })
                .collect()
        })
        .collect()
}

/// Planes with `junk` in every position and lane that is not a problem.
fn planes(columns: &[Vec<u32>], n: usize, junk: u32) -> (Vec<u32>, [usize; 2]) {
    let [rows, cols] = plane_dims(columns.len(), n);
    let mut bits = vec![junk; rows * cols];
    for (q, col) in columns.iter().enumerate() {
        for (e, &x) in col.iter().enumerate() {
            let [r, c] = plane_coord(q, e);
            bits[r * cols + c] = x;
        }
    }
    (bits, [rows, cols])
}

#[test]
fn the_sort_kernel_is_stable_and_exact_on_the_device() {
    tt_ttsim::fork_scope(|| {
        session!(s);
        let mut seed = 135u64;
        for elem in [Elem::F32, Elem::I32] {
            // 70 problems: three tile column blocks, the last ragged. Axes
            // within a tile, across two, three (a padded power of two of four)
            // and five tiles.
            for (indices, ns) in [
                (true, vec![1, 7, 32, 33, 100]),
                (false, vec![32, 64, 65, 150]),
            ] {
                for n in ns {
                    for descending in [false, true] {
                        let spec = Spec {
                            elem,
                            n,
                            descending,
                            indices,
                        };
                        let cols = problems(elem, 70, n, lcg(&mut seed) as u64);
                        let (bits, [rows, ncols]) = planes(&cols, n, 0xdead_beef);
                        let input = s.upload_bits(&bits, rows, ncols, elem).unwrap();
                        let out = s.sort_planes(&input, spec).unwrap();
                        let keys = s.download_bits(&out.keys).unwrap();
                        let idx = out.indices.as_ref().map(|t| s.download_bits(t).unwrap());
                        assert_eq!(s.download_bits(&input).unwrap(), bits, "input kept");
                        let mut moved = 0;
                        for (q, col) in cols.iter().enumerate() {
                            let (want_keys, want_idx) = reference(elem, col, descending);
                            let got_keys: Vec<u32> = (0..n)
                                .map(|e| {
                                    let [r, c] = plane_coord(q, e);
                                    keys[r * ncols + c]
                                })
                                .collect();
                            assert_eq!(got_keys, want_keys, "{spec:?} problem {q} keys");
                            if let Some(idx) = &idx {
                                let got_idx: Vec<u32> = (0..n)
                                    .map(|e| {
                                        let [r, c] = plane_coord(q, e);
                                        idx[r * ncols + c]
                                    })
                                    .collect();
                                assert_eq!(got_idx, want_idx, "{spec:?} problem {q} indices");
                            }
                            moved += usize::from(want_keys != *col);
                        }
                        assert!(
                            n == 1 || moved > 0,
                            "{spec:?}: an identity sort proves nothing"
                        );
                        s.free(out.keys).unwrap();
                        if let Some(i) = out.indices {
                            s.free(i).unwrap();
                        }
                        s.free(input).unwrap();
                    }
                }
            }
        }
    })
    .unwrap();
}

#[test]
fn the_sort_kernel_refuses_what_it_cannot_do_by_name() {
    tt_ttsim::fork_scope(|| {
        session!(s);
        let cols = problems(Elem::F32, 4, 33, 1);
        let (bits, [rows, ncols]) = planes(&cols, 33, 0);
        let input = s.upload_bits(&bits, rows, ncols, Elem::F32).unwrap();
        let spec = |elem, n, indices| Spec {
            elem,
            n,
            descending: false,
            indices,
        };
        // Too long: the message names the length and the bound.
        let e = s
            .sort_planes(&input, spec(Elem::F32, 1025, true))
            .err()
            .expect("1025")
            .to_string();
        assert!(e.contains("1025") && e.contains("1024"), "{e}");
        // The wrong element type for the planes.
        let e = s
            .sort_planes(&input, spec(Elem::I32, 33, false))
            .err()
            .expect("elem mismatch")
            .to_string();
        assert!(e.contains("sort"), "{e}");
        // Planes of the wrong height.
        let e = s
            .sort_planes(&input, spec(Elem::F32, 100, false))
            .err()
            .expect("height")
            .to_string();
        assert!(e.contains("planes"), "{e}");
        s.free(input).unwrap();
    })
    .unwrap();
}

/// The one instruction mode the sort adds to the SFPU repertoire: `SFPSWAP`
/// `MOD1_SWAP` (an unconditional exchange of two registers) under a lane
/// predicate, held to the interpreter and to its definition. A silicon
/// request's first probe: documented (`SFPSWAP.md`), the other mode in use
/// (min/max) already measured.
mod swap_probe {
    use tt_isa::backend::{self, Before, ConfigWords};
    use tt_isa::isa::generated::encode;
    use tt_isa::isa::Instruction;
    use tt_isa::tile::{L1Format, TileImage};
    use tt_kernels::datapath::{
        config_program, pack_tile_from_dst, state_id, thread_config, tile_descriptor,
        tile_unpack_config, unpack_tile_to_dst, OUT, STAGE,
    };
    use tt_kernels::sfpu::interp::Vector;
    use tt_kernels::sfpu::{Cond, Format, LReg, LoopPolicy, Program};
    use tt_tests::harness::{self, Roles, Run};

    const STAGE_B: u64 = STAGE + tt_isa::dm::TILE_SLOT;

    fn image(datums: &[u32]) -> Vec<u8> {
        let img = TileImage::new(tile_descriptor(), L1Format::Fp32).unwrap();
        let mut b = vec![0u8; img.total_bytes()];
        for (i, d) in datums.iter().enumerate() {
            let at = img.datum_bit_offset(i) / 8;
            b[at..at + 4].copy_from_slice(&d.to_le_bytes());
        }
        b
    }

    fn on_device(
        dev: &mut harness::Dev<'_>,
        math: &[Instruction],
        a: &[u32],
        b: &[u32],
    ) -> Vec<u32> {
        let mut unpack = thread_config();
        let mut words = ConfigWords::new();
        tile_unpack_config(&mut words, STAGE);
        unpack.extend(config_program(&words));
        unpack.extend(unpack_tile_to_dst(STAGE, 0));
        unpack.extend(unpack_tile_to_dst(STAGE_B, 64));
        unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
        let mut m = vec![state_id()];
        m.extend_from_slice(math);
        m.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
        let mut pack = vec![state_id()];
        pack.extend(pack_tile_from_dst(OUT, 128));
        pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
        let (ia, ib) = (image(a), image(b));
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &m,
                pack: &pack,
            })
            .stage(&[(STAGE, &ia), (STAGE_B, &ib), (OUT, &[0xA5u8; 4096])])
            .dump_rows(0)
            .read_back(&[(OUT, 4096)]),
        );
        out.l1[0]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn words(seed: u32) -> Vec<u32> {
        let specials: [u32; 12] = [
            0x0000_0000,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_1234,
            0x0000_0001,
            0x8000_0001,
            0x7fff_ffff,
            0xffff_ffff,
            0x3f80_0000,
            0xbf80_0000,
        ];
        let mut s = seed | 1;
        (0..1024)
            .map(|i| {
                if i % 3 == 0 {
                    return specials[(i / 3 + seed as usize) % specials.len()];
                }
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                s
            })
            .collect()
    }

    /// Words as `SignMagIsSmaller` orders them: the key `k = x ^ ((x >> 31) >> 1)`
    /// as a signed integer.
    fn smaller(a: u32, b: u32) -> bool {
        let key = |x: u32| (x ^ (((x as i32) >> 31) as u32 >> 1)) as i32;
        key(a) < key(b)
    }

    #[test]
    fn a_predicated_unconditional_swap_exchanges_exactly_the_lanes_it_names() {
        let (a, b) = (words(3), words(5));
        for second in [false, true] {
            let mut p = Program::with_policy(LoopPolicy::Unrolled);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, o);
                p.load(LReg::L1, Format::Int32, 64 + o);
                p.if_(Cond::Less(LReg::L0, LReg::L1), |p| {
                    p.raw(encode::sfpswap(LReg::L1.index(), LReg::L0.index(), 0).unwrap());
                });
                p.store(
                    if second { LReg::L1 } else { LReg::L0 },
                    Format::Int32,
                    128 + o,
                );
            });
            let math = p.finish();
            let mut v = Vector::new();
            v.put_tile(0, &a);
            v.put_tile(64, &b);
            v.run(&math).unwrap();
            let model = v.tile(128);
            // Its definition: where A < B the registers exchange.
            let want: Vec<u32> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| match (smaller(x, y), second) {
                    (true, false) => y,
                    (true, true) => x,
                    (false, false) => x,
                    (false, true) => y,
                })
                .collect();
            assert_eq!(model, want, "interpreter, second={second}");
            assert_ne!(want, if second { b.clone() } else { a.clone() });
            harness::in_device(|dev| {
                let got = on_device(dev, &math, &a, &b);
                assert_eq!(got, want, "device, second={second}");
            });
        }
    }
}
