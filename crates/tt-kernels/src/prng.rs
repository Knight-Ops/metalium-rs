//! Native seeded random on the device (hardware-coverage S7).
//!
//! # What the hardware gives
//!
//! `VectorUnit.md` "PRNG": every one of the 32 SFPU lanes owns a 32-bit LFSR,
//! `state' = (!popcount(state & 0x8020_0003) << 31) | state >> 1`
//! (`tt_isa::numerics::stochastic::advance`; the XNOR makes all-ones the
//! absorbing state), and `SFPMOV` mode 8 returns the *old* state and advances
//! only the enabled lanes. Its statistical properties are poor, and the lanes
//! are **not independent**: they are one sequence read at different times.
//!
//! # What is known about seeding (`step91_seeded_prng`, both cards)
//!
//! * A `WRCFG` to `PRNG_SEED_Seed_Val` does not restart silicon's stream. A
//!   full-width RISC-V store to the same word, a fence and 512 NOP iterations
//!   does, on both cards (`tt_isa::dataflow::SEED_SETTLE_NOPS`). The role
//!   firmware does exactly that for a program that opens with a seed
//!   directive (`tt_isa::dataflow::seed_directive`).
//! * After the restart, **ttsim**'s lane `i` holds `advance^(96 - 2 i)(seed)`
//!   (fitted here on seeds 0, 1, 0x1234_5678, 0x8000_0000, 0x5555_5555,
//!   0xaaaa_aaaa: every one exact). So lane `i + 1` is lane `i`'s state two
//!   steps *earlier*: `lane[i + 1] = lane[i] << 2 | two more bits`, the
//!   adjacent-lane correlation step91 asserts.
//! * **Silicon** is the same shape shifted by one lane: seed 0 gives lane 0
//!   `0xf173cc27 = advance^98(0)` and lane 1 `0xc5cf309e = advance^96(0)`, ttsim's
//!   lane 0. The model's silicon constant is therefore `98 - 2 i`, established
//!   for seed 0 lanes 0 and 1 only and **PENDING MEASUREMENT** for the other
//!   lanes and seeds (probe: `step140_prng_model::silicon_lane_initialisation`,
//!   which dumps all 32 lanes for 8 seeds and compares them with the model).
//!
//! # The generator built on it
//!
//! Using lane words directly would give a tile in which adjacent lanes are
//! shifts of each other and adjacent reads are halvings (`raw` in
//! `step140_prng_model`, which the statistical gate watches fail). So:
//!
//! 1. one tile is one program run seeded by the host: `seed32 =
//!    splitmix64(splitmix64(base ^ role) + tile * golden)` (high half), so a
//!    tile's bits depend on `(base, role, tile)` only -- not on which core runs
//!    it, how many cores there are, the run decomposition or the card;
//! 2. each of the 32 store slots reads the PRNG [`READS_PER_GROUP`] = 33 times
//!    and keeps the last (33 is odd, so no two of the 1024 words of a tile come
//!    from one stream position: collisions need `2 (i - i') = 33 (g - g')`);
//! 3. a bijective ARX mixer ([`mix32`]: six xor-right-shifts, five
//!    add-left-shifts) removes the shift structure -- the pair tests of
//!    `step140` pass it and fail the raw words and a weaker mixer;
//! 4. a uniform is `(0x3f80_0000 | word >> 9) - 1.0`: 23 random bits, exactly
//!    `k 2^-23`, `k in [0, 2^23)`, so `[0, 1 - 2^-23]` with no rounding.
//!
//! Honest limits: a tile carries at most 32 bits of entropy (the seed), and the
//! mixer is an empirical decorrelator, not a cryptographic one. This is a
//! dropout/initialisation generator, not a statistical-test-suite one.
//!
//! Results are reproducible per target ([`Target`]): the one-lane offset above
//! makes ttsim and silicon draw different (equally good) streams.

use std::sync::Arc;
use tt_device::Transport;
use tt_isa::dataflow::seed_directive;
use tt_isa::dm::record;
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::numerics::stochastic::advance;

use crate::code::Loop;
use crate::session::Session;
use crate::sfpu::kernel::{plan_layout, roles_code, Layout, Operands, OUT_ROW};
use crate::sfpu::{Format, LReg, Program};
use crate::tensor::{DramTensor, Elem, Pad, Step, TensorError};

/// SFPU lanes.
pub const LANES: usize = 32;
/// Store slots (`SFPSTORE`s) a tile takes: each fills 32 of its 1024 datums.
pub const GROUPS: usize = 32;
/// PRNG reads per slot; the last is kept.
pub const READS_PER_GROUP: usize = 33;
/// Mixer right shifts, in the order [`mix32`] applies them, and the left
/// shifts between them.
const RIGHT: [u32; 6] = [16, 6, 11, 13, 17, 11];
const LEFT: [u32; 5] = [10, 3, 15, 7, 5];

/// Which machine's lane initialisation the model follows.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// The pinned simulator: lane `i` starts `96 - 2 i` steps from the seed.
    Ttsim,
    /// Blackhole silicon: one lane further, `98 - 2 i` (seed 0, lanes 0 and 1
    /// measured; PENDING MEASUREMENT beyond them).
    Silicon,
}

impl Target {
    pub const fn from_simulated(simulated: bool) -> Self {
        if simulated {
            Target::Ttsim
        } else {
            Target::Silicon
        }
    }

    /// LFSR steps from the (restarted) seed register to lane `lane`'s first
    /// read.
    pub const fn initial_steps(self, lane: usize) -> usize {
        let base = match self {
            Target::Ttsim => 96,
            Target::Silicon => 98,
        };
        base - 2 * lane
    }
}

/// One step of splitmix64 (Steele, Lea & Flood; Vigna's constants).
pub const fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The PRNG seed of tile `tile` of the draw `(base, role)`. Never the
/// absorbing all-ones state.
pub const fn tile_seed(base: u64, tile: u64, role: u32) -> u32 {
    let stream = splitmix64(base ^ ((role as u64) << 32 | 0x5EED));
    let mixed = splitmix64(stream.wrapping_add(tile.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let seed = (mixed >> 32) as u32;
    if seed == u32::MAX {
        mixed as u32 ^ 0x5EED_5EED
    } else {
        seed
    }
}

/// The mixer: [`RIGHT`] and [`LEFT`] interleaved, `x ^= x >> r` then
/// `x += x << l`. A bijection of `u32`.
pub const fn mix32(mut x: u32) -> u32 {
    let mut i = 0;
    while i < 5 {
        x ^= x >> RIGHT[i];
        x = x.wrapping_add(x << LEFT[i]);
        i += 1;
    }
    x ^ (x >> RIGHT[5])
}

/// Lane `lane`'s state straight after a restart from `seed`: what a dump
/// before any read shows (`step91`'s snapshots).
pub fn initial_state(seed: u32, target: Target, lane: usize) -> u32 {
    (0..target.initial_steps(lane)).fold(seed, |s, _| advance(s))
}

/// The raw lane words of a restarted seed: `result[g][lane]` is what read
/// number `READS_PER_GROUP * g + READS_PER_GROUP - 1` returns (the unmixed word
/// of store slot `g`), straight from the LFSR.
pub fn raw_words(seed: u32, target: Target) -> [[u32; LANES]; GROUPS] {
    let longest = target.initial_steps(0) + READS_PER_GROUP * GROUPS;
    let mut sequence = Vec::with_capacity(longest);
    let mut s = seed;
    for _ in 0..longest {
        sequence.push(s);
        s = advance(s);
    }
    std::array::from_fn(|g| {
        std::array::from_fn(|lane| {
            sequence[target.initial_steps(lane) + READS_PER_GROUP * g + READS_PER_GROUP - 1]
        })
    })
}

/// Where store slot `slot`'s lane `lane` lands in a tile image (face order,
/// `Dst` row `r` datums `16 r..16 r + 16`): lane `8 j + c` is row `4 (slot/2) + j`,
/// column `2 c + slot % 2` (`sfpu::sort`'s plane geometry; `SFPLOAD.md`).
pub const fn image_index(slot: usize, lane: usize) -> usize {
    (4 * (slot / 2) + lane / 8) * 16 + (lane % 8) * 2 + slot % 2
}

/// Tile-image index to `(row, column)` of the 32x32 tile (four 16x16 faces,
/// row-major).
pub const fn tile_coord(i: usize) -> (usize, usize) {
    let face = i / 256;
    ((face / 2) * 16 + (i % 256) / 16, (face % 2) * 16 + i % 16)
}

/// The 1024 mixed words of a tile, in tile-image order: what [`Output::Word`]
/// stores.
pub fn tile_words(seed: u32, target: Target) -> Vec<u32> {
    let raw = raw_words(seed, target);
    let mut image = vec![0u32; 1024];
    for (slot, row) in raw.iter().enumerate() {
        for (lane, &w) in row.iter().enumerate() {
            image[image_index(slot, lane)] = mix32(w);
        }
    }
    image
}

/// A mixed word as the uniform [`Output::Unit`] stores: `k 2^-23` from its top
/// 23 bits, exact.
pub fn unit_bits(word: u32) -> u32 {
    (f32::from_bits(0x3f80_0000 | word >> 9) - 1.0).to_bits()
}

/// A mixed word's uniform in `[0, 1 - 2^-23]`.
pub fn unit(word: u32) -> f32 {
    f32::from_bits(unit_bits(word))
}

/// The 1024 uniform bit patterns of a tile, in tile-image order.
pub fn tile_units(seed: u32, target: Target) -> Vec<u32> {
    tile_words(seed, target)
        .into_iter()
        .map(unit_bits)
        .collect()
}

/// What a draw's tile kernel stores.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Output {
    /// F32 uniforms, `k 2^-23`.
    Unit,
    /// The mixed 32-bit words, raw.
    Word,
}

/// A tile's math program: the constants, then per store slot the reads, the
/// mixer and the store (`Output`).
pub fn math(output: Output) -> crate::code::Code {
    let mut p = Program::new();
    let shifts = [LReg::L2, LReg::L3, LReg::L4, LReg::L5, LReg::L6];
    for (reg, shift) in shifts.into_iter().zip(RIGHT) {
        p.loadi_bits(reg, (shift as i32).wrapping_neg() as u32);
    }
    p.for_each_row_group(64, |p, o| {
        for _ in 1..READS_PER_GROUP {
            p.read_prng(LReg::L1);
        }
        p.read_prng(LReg::L0);
        for i in 0..5 {
            // x ^= x >> RIGHT[i]
            p.mov(LReg::L0, LReg::L1);
            p.shr_by(shifts[i], LReg::L1);
            p.xor(LReg::L1, LReg::L0);
            // x += x << LEFT[i]
            p.shl(LReg::L0, LEFT[i], LReg::L1);
            p.iadd(LReg::L1, LReg::L0);
        }
        p.mov(LReg::L0, LReg::L1);
        p.shr_by(shifts[2], LReg::L1); // RIGHT[5] == RIGHT[2] == 11
        p.xor(LReg::L1, LReg::L0);
        match output {
            Output::Word => p.store(LReg::L0, Format::Int32, OUT_ROW + o),
            Output::Unit => {
                // (0x3f80_0000 | x >> 9) - 1.0
                p.loadi_bits(LReg::L1, (-9i32) as u32);
                p.shr_by(LReg::L1, LReg::L0);
                p.loadi_bits(LReg::L1, 0x3f80_0000);
                p.or(LReg::L0, LReg::L1, LReg::L0);
                p.sub(LReg::L0, LReg::L1, LReg::L0);
                p.store(LReg::L0, Format::Fp32, OUT_ROW + o);
            }
        }
    });
    p.finish_code()
}

const _: () = assert!(RIGHT[5] == RIGHT[2]);

/// A seeded kernel's role programs and block repeats, as `Step::Kernel` holds them.
pub type SeededRoles = (Arc<[Vec<Instruction>; 3]>, Arc<[Vec<Loop>; 3]>);

/// The kernel's role programs for one tile, before a seed is added.
pub struct Kernel {
    pub layout: Layout,
    roles: [Vec<Instruction>; 3],
    loops: [Vec<Loop>; 3],
}

impl Kernel {
    pub fn new(output: Output) -> Result<Kernel, TensorError> {
        let layout =
            plan_layout(1, Operands::Unary).map_err(|e| TensorError::Shape(e.to_string()))?;
        let (roles, loops) = roles_code(&layout, Operands::Unary, &math(output));
        Ok(Kernel {
            layout,
            roles,
            loops,
        })
    }

    /// The roles with a seed directive at the head of the math role: the
    /// firmware restarts the PRNG from `seed` before pushing the rest. The two
    /// directive words are carried as NOPs in the host's instruction list
    /// (`Instruction::new` with `NOP`'s definition) and are never pushed.
    pub fn seeded(&self, seed: u32) -> SeededRoles {
        let nop = encode::nop().unwrap();
        let words = seed_directive(seed);
        let mut roles = self.roles.clone();
        let directive = words.map(|w| Instruction::new(w, nop.def()));
        roles[1].splice(0..0, directive);
        let mut loops = self.loops.clone();
        for l in &mut loops[1] {
            l.start += words.len() as u32;
        }
        (Arc::new(roles), Arc::new(loops))
    }
}

/// Distinct draws of one `(base)`: each role is an independent tile stream.
pub mod role {
    pub const UNIT: u32 = 0;
    pub const UNIFORM: u32 = 1;
    pub const BERNOULLI: u32 = 2;
    pub const NORMAL_RADIUS: u32 = 3;
    pub const NORMAL_ANGLE: u32 = 4;
    pub const INT_RANGE: u32 = 5;
    pub const INT_WORD: u32 = 6;
}

/// The largest range an integer `Uniform(lo, hi)` takes: the multiply-shift is
/// on the 23-bit uniform, so a larger range would leave values unreachable.
pub const INT_RANGE_MAX: u64 = 1 << 23;

/// `x` as the largest float below it (`x` finite).
pub fn next_down(x: f32) -> f32 {
    if x == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits - 1 } else { bits + 1 })
}

impl<T: Transport> Session<T> {
    /// Which model matches this session's transport.
    pub fn prng_target(&mut self) -> Target {
        Target::from_simulated(self.is_simulated())
    }

    /// A `[rows, cols]` tensor of tile kernel output: tile `t` (row-major tile
    /// order) is seeded `tile_seed(base, t, role)`. Edge-tile padding holds
    /// random words ([`Pad::Undefined`]).
    pub fn random_tiles(
        &mut self,
        dims: [usize; 2],
        output: Output,
        base: u64,
        role: u32,
    ) -> Result<DramTensor, TensorError> {
        if dims[0] == 0 || dims[1] == 0 {
            return Err(TensorError::Shape(format!(
                "random tensor {dims:?} is empty"
            )));
        }
        // A trace replays its jobs verbatim, seed directives included: every
        // replay would draw the same bits. Advancing a seed buffer per replay
        // needs the directive's seed read from GDDR by the firmware (not
        // built), so random inside a trace is refused, never replayed stale.
        if self.capturing() {
            return Err(TensorError::Shape(
                "random inside a trace is refused: replay would repeat this draw's seeds \
                 (draw the tensor before the trace and pass it in)"
                    .into(),
            ));
        }
        let kernel = Kernel::new(output)?;
        let elem = match output {
            Output::Unit => Elem::F32,
            Output::Word => Elem::I32,
        };
        let out = DramTensor::alloc_elem(self.dram_alloc()?, dims[0], dims[1], elem)?;
        let [rt, ct] = out.grid();
        let reference = out.tensor_ref().encode();
        let layout = &kernel.layout;
        let mut jobs = Vec::with_capacity(rt * ct);
        for tile in 0..rt * ct {
            let (roles, loops) = kernel.seeded(tile_seed(base, tile as u64, role));
            // The kernel unpacks an A tile it does not use: gather the (still
            // unwritten) output tile, so the streaming credits stay balanced.
            let gather = vec![
                [
                    record::READ_RUN,
                    tile as u32,
                    1,
                    layout.a_at as u32,
                    0,
                    ct as u32,
                    0,
                    0,
                ],
                reference[0],
                reference[1],
            ];
            let scatter = vec![
                [
                    record::WRITE_RUN,
                    tile as u32,
                    1,
                    layout.out_at as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                reference[0],
                reference[1],
            ];
            jobs.push(vec![
                Step::List {
                    what: "random gather",
                    entries: gather,
                },
                Step::Kernel {
                    roles,
                    init: layout.init.clone(),
                    mop: Box::new([None; 3]),
                    loops,
                    half: None,
                },
                Step::List {
                    what: "random scatter",
                    entries: scatter,
                },
            ]);
        }
        out.set_pad(Pad::Undefined);
        if let Err(e) = self.submit_jobs(jobs, crate::session::RESET_BUDGET) {
            self.dram_alloc()?.free(&out.placement);
            return Err(e);
        }
        Ok(out)
    }

    /// One element-wise step, freeing its input.
    fn random_step(
        &mut self,
        kind: u32,
        scalar: f32,
        a: DramTensor,
        b: Option<&DramTensor>,
    ) -> Result<DramTensor, TensorError> {
        let op = crate::tensor::Eltwise {
            kind,
            scalar,
            scalar2: 0.0,
        };
        let out = self.eltwise(op, &a, b);
        self.free(a)?;
        out
    }

    /// `lo + (hi - lo) u` for the unit uniform `u`, in `[lo, hi)`: two F32
    /// roundings (`s = fl(hi - lo)`, `fl(fl(u s) + lo)`), then a clamp to the
    /// float below `hi` so the half-open interval is exact. Within
    /// [`uniform_bound`] of the real affine map.
    pub fn random_uniform(
        &mut self,
        dims: [usize; 2],
        base: u64,
        lo: f32,
        hi: f32,
    ) -> Result<DramTensor, TensorError> {
        use crate::kind;
        use crate::sfpu::ops::kind_sfpu;
        let scale = hi - lo;
        if !(lo.is_finite() && hi.is_finite() && lo < hi && scale.is_finite() && scale > 0.0) {
            return Err(TensorError::Shape(format!(
                "uniform [{lo}, {hi}) is not a finite, nonempty interval"
            )));
        }
        let u = self.random_tiles(dims, Output::Unit, base, role::UNIFORM)?;
        let t = self.random_step(kind::MUL_SCALAR, scale, u, None)?;
        let w = self.random_step(kind::ADD_SCALAR, lo, t, None)?;
        self.random_step(kind_sfpu::CLAMP_MAX, next_down(hi), w, None)
    }

    /// `1.0` where `u < p`, else `0.0` (`as_int`: `I32` `1`/`0`), for the unit
    /// uniform `u`: the probability of one is `ceil(p 2^23) / 2^23`, within
    /// `2^-23` above `p`.
    pub fn random_bernoulli(
        &mut self,
        dims: [usize; 2],
        base: u64,
        p: f32,
        as_int: bool,
    ) -> Result<DramTensor, TensorError> {
        use crate::sfpu::ops::kind_sfpu;
        if !(0.0..=1.0).contains(&p) {
            return Err(TensorError::Shape(format!(
                "Bernoulli probability {p} is not in [0, 1]"
            )));
        }
        let u = self.random_tiles(dims, Output::Unit, base, role::BERNOULLI)?;
        let mask = self.random_step(kind_sfpu::LT_S, p, u, None)?;
        self.random_step(
            if as_int {
                kind_sfpu::BOOL_TO_I32
            } else {
                kind_sfpu::BOOL_TO_F32
            },
            0.0,
            mask,
            None,
        )
    }

    /// `mean + std z`, `z = sqrt(-2 ln(1 - u1)) cos(2 pi u2)` (Box-Muller, the
    /// cosine half) from two independent unit draws, on the existing
    /// `LOG`/`SQRT`/`COS` programs; `1 - u1` is exact and in `(0, 1]`, so the
    /// tail is cut at `sqrt(-2 ln 2^-23) = 5.64` sigma.
    pub fn random_normal(
        &mut self,
        dims: [usize; 2],
        base: u64,
        mean: f32,
        std: f32,
    ) -> Result<DramTensor, TensorError> {
        use crate::kind;
        use crate::sfpu::ops::kind_sfpu;
        if !(mean.is_finite() && std.is_finite() && std >= 0.0) {
            return Err(TensorError::Shape(format!(
                "normal(mean {mean}, std {std}) needs finite mean and std >= 0"
            )));
        }
        let u1 = self.random_tiles(dims, Output::Unit, base, role::NORMAL_RADIUS)?;
        let neg = self.random_step(kind::MUL_SCALAR, -1.0, u1, None)?;
        let x = self.random_step(kind::ADD_SCALAR, 1.0, neg, None)?;
        let l = self.random_step(kind_sfpu::LOG, 0.0, x, None)?;
        let r2 = self.random_step(kind::MUL_SCALAR, -2.0, l, None)?;
        let radius = self.random_step(kind_sfpu::SQRT, 0.0, r2, None)?;
        let u2 = self.random_tiles(dims, Output::Unit, base, role::NORMAL_ANGLE)?;
        let theta = self.random_step(kind::MUL_SCALAR, TWO_PI, u2, None)?;
        let c = self.random_step(kind_sfpu::COS, 0.0, theta, None)?;
        let z = self.eltwise(
            crate::tensor::Eltwise {
                kind: kind::MUL,
                scalar: 0.0,
                scalar2: 0.0,
            },
            &radius,
            Some(&c),
        );
        self.free(radius)?;
        self.free(c)?;
        let z = z?;
        let scaled = self.random_step(kind::MUL_SCALAR, std, z, None)?;
        self.random_step(kind::ADD_SCALAR, mean, scaled, None)
    }

    /// [`Session::random_normal`] truncated toward zero to `I32` (saturating),
    /// as Rust's `f64 as i32` and so Burn's integer `Distribution::Normal`.
    pub fn random_normal_int(
        &mut self,
        dims: [usize; 2],
        base: u64,
        mean: f32,
        std: f32,
    ) -> Result<DramTensor, TensorError> {
        let f = self.random_normal(dims, base, mean, std)?;
        self.random_step(crate::sfpu::ops::kind_sfpu::F32_TO_I32, 0.0, f, None)
    }

    /// Integers in `[lo, hi)`: `lo + floor(u (hi - lo))`, `u` the unit uniform.
    /// The 2^23 equally likely `u` values split among `R = hi - lo` results as
    /// evenly as integers allow: every result has `floor(2^23 / R)` or
    /// `ceil(2^23 / R)` of them, so the modulo bias is at most `R / 2^23`
    /// relative ([`int_range_bias`]); `R` is limited to [`INT_RANGE_MAX`].
    pub fn random_int_range(
        &mut self,
        dims: [usize; 2],
        base: u64,
        lo: i32,
        hi: i32,
    ) -> Result<DramTensor, TensorError> {
        use crate::sfpu::ops::kind_sfpu;
        let range = i64::from(hi) - i64::from(lo);
        if range < 1 || range as u64 > INT_RANGE_MAX {
            return Err(TensorError::Shape(format!(
                "integer uniform [{lo}, {hi}): a range of {range} outside 1..={INT_RANGE_MAX}"
            )));
        }
        let u = self.random_tiles(dims, Output::Unit, base, role::INT_RANGE)?;
        let scaled = self.random_step(crate::kind::MUL_SCALAR, range as f32, u, None)?;
        let floor = self.random_step(kind_sfpu::F32_TO_I32, 0.0, scaled, None)?;
        self.random_step(kind_sfpu::INT_ADD_S, f32::from_bits(lo as u32), floor, None)
    }

    /// Integers uniform over all 32 bits (Burn's `Distribution::Default` for
    /// `i32`): the mixed words, raw.
    pub fn random_int_words(
        &mut self,
        dims: [usize; 2],
        base: u64,
    ) -> Result<DramTensor, TensorError> {
        self.random_tiles(dims, Output::Word, base, role::INT_WORD)
    }
}

/// `2 pi` as the F32 the program multiplies by.
pub const TWO_PI: f32 = std::f32::consts::TAU;

/// The most any integer value of `random_int_range(R)` is over- or
/// under-represented relative to `2^23 / R`: counts are `floor` or `ceil`.
pub fn int_range_bias(range: u64) -> f64 {
    range as f64 / (1u64 << 23) as f64
}

/// What an element of `random_uniform(lo, hi)` may differ from the real
/// `lo + u (hi - lo)` by: the scale's rounding (`<= ulp(s) / 2`, times `u < 1`),
/// the product's (`<= 2^-24 |u s|`), the sum's (`<= 2^-24 |r|`), and the clamp
/// (zero or one ulp of `hi`, only where the sum rounded up to `hi`).
pub fn uniform_bound(lo: f32, hi: f32) -> f64 {
    let s = f64::from(hi) - f64::from(lo);
    let ulp = |x: f64| f64::from(f32::from_bits((x.abs() as f32).to_bits() + 1)) - x.abs();
    let top = f64::from(lo.abs()).max(f64::from(hi.abs()));
    0.5 * ulp(s) + s / 16_777_216.0 + top / 16_777_216.0 + ulp(top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfpu::interp::Vector;

    fn interpreted(output: Output, seed: u32, target: Target) -> Vec<u32> {
        let mut v = Vector::new();
        v.prng = Some(std::array::from_fn(|lane| {
            (0..target.initial_steps(lane)).fold(seed, |s, _| advance(s))
        }));
        v.run(&math(output).expand()).unwrap();
        v.tile(OUT_ROW as usize)
    }

    #[test]
    fn the_model_matches_the_interpreted_program() {
        for target in [Target::Ttsim, Target::Silicon] {
            for seed in [0, 1, 0x1234_5678, 0x8000_0000, 0xdead_beef, 0xffff_fffe] {
                assert_eq!(
                    interpreted(Output::Word, seed, target),
                    tile_words(seed, target),
                    "{target:?} {seed:#x} words"
                );
                assert_eq!(
                    interpreted(Output::Unit, seed, target),
                    tile_units(seed, target),
                    "{target:?} {seed:#x} units"
                );
            }
        }
    }

    #[test]
    fn lane_initialisation_matches_the_measured_ttsim_and_silicon_seed_zero() {
        let state = |t: Target, lane| (0..t.initial_steps(lane)).fold(0, |s, _| advance(s));
        // step91: ttsim 0xc5cf309e / 0x173cc27a, silicon 0xf173cc27 / 0xc5cf309e.
        assert_eq!(
            [state(Target::Ttsim, 0), state(Target::Ttsim, 1)],
            [0xc5cf_309e, 0x173c_c27a]
        );
        assert_eq!(
            [state(Target::Silicon, 0), state(Target::Silicon, 1)],
            [0xf173_cc27, 0xc5cf_309e]
        );
    }

    #[test]
    fn mixer_is_a_bijection_on_samples_and_the_unit_is_exact() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..200_000u32 {
            assert!(seen.insert(mix32(i.wrapping_mul(2_654_435_761))));
        }
        for w in [0u32, 1, 0x1ff, 0x200, u32::MAX, 0x8000_0000] {
            let u = unit(w);
            assert!((0.0..1.0).contains(&u));
            assert_eq!(f64::from(u), f64::from(w >> 9) / 8_388_608.0);
        }
    }

    #[test]
    fn a_tile_has_no_repeated_stream_position() {
        // 1024 words, one stream position each: no two raw words alias.
        for target in [Target::Ttsim, Target::Silicon] {
            let positions: std::collections::HashSet<_> = (0..GROUPS)
                .flat_map(|g| {
                    (0..LANES).map(move |l| {
                        target.initial_steps(l) + READS_PER_GROUP * g + READS_PER_GROUP - 1
                    })
                })
                .collect();
            assert_eq!(positions.len(), 1024);
        }
    }

    #[test]
    fn image_index_is_a_permutation_and_seeds_depend_on_every_input() {
        let all: std::collections::HashSet<_> = (0..GROUPS)
            .flat_map(|g| (0..LANES).map(move |l| image_index(g, l)))
            .collect();
        assert_eq!(all.len(), 1024);
        assert!(all.iter().all(|&i| i < 1024));
        let s = tile_seed(1, 2, 3);
        assert_ne!(s, tile_seed(2, 2, 3));
        assert_ne!(s, tile_seed(1, 3, 3));
        assert_ne!(s, tile_seed(1, 2, 4));
        assert_ne!(tile_seed(0, 0, 0), u32::MAX);
        // splitmix64's published first output for seed 0.
        assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn the_seed_directive_heads_only_the_math_role() {
        let k = Kernel::new(Output::Unit).unwrap();
        let (roles, loops) = k.seeded(0xabcd_1234);
        assert_eq!(roles[1][0].word(), tt_isa::dataflow::SEED_DIRECTIVE);
        assert_eq!(roles[1][1].word(), 0xabcd_1234);
        assert_eq!(roles[0], k.roles[0]);
        assert_eq!(roles[2], k.roles[2]);
        assert_eq!(roles[1].len(), k.roles[1].len() + 2);
        for (a, b) in loops[1].iter().zip(&k.loops[1]) {
            assert_eq!(a.start, b.start + 2);
        }
        let stored = crate::code::Code {
            ins: roles[1].clone(),
            loops: loops[1].clone(),
        }
        .stored()
        .unwrap();
        assert!(stored.1 & tt_isa::mailbox::loops::LOOPED != 0);
    }
}
