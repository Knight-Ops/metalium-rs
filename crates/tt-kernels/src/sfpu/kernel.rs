//! The SFPU tile kernel (`hardware-coverage.md` F3): a run of tiles through
//! `Dst`, an SFPU program over each.
//!
//! Three roles, as LLK splits them: T0 unpacks each tile's operands into
//! `Dst` -- `A` to rows 0..64, `B` to rows 64..128 -- T1 runs the op's program,
//! which writes its result to rows 128..192, and T2 packs those rows to the
//! tile's output slot. The data mover gathers the run's operands into L1
//! before the kernel and scatters the outputs after it
//! (`tt_isa::dm::record::{READ_RUN, WRITE_RUN}`), so the kernel sees only L1.
//!
//! The roles hand each tile on through three semaphores, declared through
//! `crate::l1` like every kernel's: `unpacked` (T0 -> T1), `computed` (T1 ->
//! T2) and `free` (T2 -> T0: the packer has read `Dst`, the next tile may be
//! unpacked over it). The consumer waits -- `STALLWAIT` blocks the
//! *consumer*'s unit (divergence row 46) -- and each producer posts only once
//! its unit has finished. A run takes every one it posts, so it leaves them as
//! it found them (`unpacked` and `computed` at zero, `free` at one) and the
//! mover can run kernels back to back (`Kernel::restores_semaphores`).
//!
//! One tile at a time: the three roles overlap across a tile's stages only in
//! that each starts its next tile's setup while waiting. Double-buffering
//! `Dst` is 9.8's.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::dm::TILE_SLOT;
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Semaphore, Unit};

use crate::datapath::{
    config_program, pack_tile_from_dst, state_id, thread_config, tile_unpack_config,
    unpack_datums_to_dst, unpack_tile_to_dst,
};
use crate::l1::{PlanError, Requirements};
use crate::runtime::SemaphoreInit;

/// Where an op's operands are in `Dst`, and its result.
pub const A_ROW: u32 = 0;
pub const B_ROW: u32 = 64;
pub const OUT_ROW: u32 = 128;

/// What a tile's second operand is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operands {
    /// `A` only.
    Unary,
    /// `A` and `B`, the same shape.
    Binary,
    /// `A`, and `B`'s first row broadcast down every row: `B`'s row 0 is laid
    /// into `Dst` four times over, its first face's sixteen columns in rows
    /// `B_ROW..B_ROW + 4` and its second face's in `B_ROW + 4..B_ROW + 8`, so
    /// one `SFPLOAD` of either group gives every lane its column's value
    /// (`Program`s read them with [`bias_row`]).
    RowBroadcast,
}

/// The `Dst` row of the broadcast row's group for row group `g` of a tile:
/// faces 0 and 2 (row groups 0..4 and 8..12) take columns 0..16, faces 1 and 3
/// columns 16..32.
pub const fn bias_row(g: u32) -> u32 {
    if (g / 4) % 2 == 0 {
        B_ROW
    } else {
        B_ROW + 4
    }
}

/// The kernel's three semaphores, as a plan placed them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SfpuSemaphores {
    pub unpacked: Semaphore,
    pub computed: Semaphore,
    pub free: Semaphore,
}

/// Where a run of `tiles` tiles lives in L1, and its semaphores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub tiles: usize,
    /// The first operand's slots, one per tile.
    pub a_at: u64,
    /// The second operand's, unless unary.
    pub b_at: Option<u64>,
    /// The packer's output slots: datums at `TILE_DATA` past each slot's start.
    pub out_at: u64,
    pub sems: SfpuSemaphores,
    pub init: Vec<SemaphoreInit>,
}

/// Plan a run of `tiles` tiles in the data arena: slots for the operands and
/// the outputs, and the three semaphores.
pub fn plan_layout(tiles: usize, operands: Operands) -> Result<Layout, PlanError> {
    let mut req = Requirements::new(1);
    let bytes = tiles as u64 * TILE_SLOT;
    let align = tt_isa::dram::ALIGN;
    let a = req.scratch("sfpu A slots", bytes, align, 0..1);
    let b = (operands != Operands::Unary).then(|| req.scratch("sfpu B slots", bytes, align, 0..1));
    let out = req.scratch("sfpu output slots", bytes, align, 0..1);
    // Declared in this order so the planner numbers them as a matmul's are --
    // the first starting at zero, the second at one -- and a tile that ran
    // one needs no setup run before the other (`runtime::Resident`).
    let unpacked = req.semaphore("sfpu unpacked", 0, 0..1);
    let free = req.semaphore("sfpu Dst free", 1, 0..1);
    let computed = req.semaphore("sfpu computed", 0, 0..1);
    let plan = req.plan(tt_isa::l1::DATA)?;
    Ok(Layout {
        tiles,
        a_at: plan.addr(a),
        b_at: b.map(|b| plan.addr(b)),
        out_at: plan.addr(out),
        sems: SfpuSemaphores {
            unpacked: plan.semaphore(unpacked),
            computed: plan.semaphore(computed),
            free: plan.semaphore(free),
        },
        init: plan.semaphore_init(),
    })
}

/// Most tiles a run may have: three slots each in the data arena.
pub fn max_tiles(operands: Operands) -> usize {
    let per = if operands == Operands::Unary { 2 } else { 3 };
    (tt_isa::l1::DATA.len() / (per * TILE_SLOT)) as usize
}

/// The three role programs of a run over `layout`'s tiles, `math` -- one
/// tile's SFPU program, reading `A_ROW` (and `B_ROW`) and writing `OUT_ROW`
/// -- run once per tile.
pub fn roles(layout: &Layout, operands: Operands, math: &[Instruction]) -> [Vec<Instruction>; 3] {
    let s = layout.sems;
    let slot = |base: u64, n: usize| base + n as u64 * TILE_SLOT;

    // Everything the unpacker's address generators read, set here: a kernel
    // inherits the thread's ADCs from whatever ran before (a matmul leaves Z
    // at its last face), and a stale counter moves where a tile is read from
    // (`tt-metal-concepts-review.md` G11).
    let mut unpack = thread_config();
    unpack.extend(crate::datapath::clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, layout.a_at);
    unpack.extend(config_program(&words));
    let mut m = vec![state_id()];
    let mut pack = vec![state_id()];
    for n in 0..layout.tiles {
        unpack.extend(sync::take(s.free, Before::UNPACKER));
        unpack.extend(unpack_tile_to_dst(slot(layout.a_at, n), A_ROW));
        match (operands, layout.b_at) {
            (Operands::Binary, Some(b)) => {
                unpack.extend(unpack_tile_to_dst(slot(b, n), B_ROW));
            }
            (Operands::RowBroadcast, Some(b)) => {
                // Row 0 of faces 0 and 1: datums 0..16 and 256..272.
                for (face, first) in [(0, 0), (1, 256)] {
                    for r in 0..4 {
                        unpack.extend(unpack_datums_to_dst(
                            slot(b, n),
                            first,
                            16,
                            B_ROW + 4 * face + r,
                        ));
                    }
                }
            }
            (Operands::Unary, _) => {}
            _ => panic!("{operands:?} with no B slots"),
        }
        unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));

        m.extend(sync::take(s.unpacked, Before::SFPU));
        m.extend_from_slice(math);
        m.extend(sync::post_after(Unit::Sfpu, s.computed));

        pack.extend(sync::take(s.computed, Before::PACKER));
        pack.extend(pack_tile_from_dst(
            slot(layout.out_at, n) + tt_isa::dm::TILE_DATA,
            OUT_ROW,
        ));
        pack.extend(sync::post_after(Unit::Packer, s.free));
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, m, pack]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SFPU kernel's semaphores agree with a matmul's on every one they
    /// share, so the two alternate on a tile with no setup between them.
    #[test]
    fn the_semaphores_agree_with_a_matmul_s() {
        let (_, matmul) = crate::matmul::MatmulSemaphores::alone();
        for operands in [Operands::Unary, Operands::Binary, Operands::RowBroadcast] {
            let sfpu = plan_layout(8, operands).unwrap().init;
            for m in &matmul {
                assert!(
                    sfpu.iter().all(|s| s.0 != m.0 || s == m),
                    "{operands:?}: {sfpu:?} against {matmul:?}"
                );
            }
        }
    }
}
