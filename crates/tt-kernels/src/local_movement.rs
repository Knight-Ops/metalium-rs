//! Exclusive T0 L1 movement, carried by ordinary serialized ownership jobs.
//! B/NC use software NoC movers. No firmware/host TDMA-RISC users exist in this
//! stack; bring-up completes before these jobs. Never opt into staging overlap.
use crate::{
    l1::{Buf, Plan, Requirements},
    tensor::{Job, Placement, Result, Step, TensorError},
};
use std::sync::Arc;
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{thcon, thread},
    dm::{op, TILE_DATA},
    isa::{generated::encode, Instruction},
    scalar::{self, OffsetHalf, OffsetIncrement, TransferWidth},
    sync::{self, Semaphore},
};

fn invalid(message: &str) -> TensorError {
    TensorError::Shape(message.into())
}
/// Reject empty public tensors and all arithmetic overflow before allocation.
pub(crate) fn geometry(dims: [usize; 2], bytes: u64) -> Result<usize> {
    let [rows, cols] = dims;
    if dims.contains(&0)
        || rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(bytes as usize))
            .is_none()
    {
        return Err(invalid("XMOV requires a nonempty, nonoverflowing shape"));
    }
    let tiles = rows
        .div_ceil(32)
        .checked_mul(cols.div_ceil(32))
        .ok_or_else(|| invalid("XMOV tile count overflow"))?;
    tiles
        .checked_mul((1024 * bytes + 64) as usize)
        .ok_or_else(|| invalid("XMOV storage overflow"))?;
    Ok(tiles)
}
struct Local {
    plan: Plan,
    source: Buf,
    dest: Buf,
    bytes: u64,
    ready: Semaphore,
    done: Semaphore,
    program: Vec<Instruction>,
}
impl Local {
    fn new(bytes: u64) -> Result<Self> {
        let mut req = Requirements::new(1);
        let source = req.scratch("XMOV source tile", bytes, 16, 0..1);
        let dest = req.scratch("XMOV destination tile", bytes, 16, 0..1);
        let ready = req.semaphore("XMOV output reserved", 0, 0..1);
        let done = req.semaphore("XMOV T0 complete", 0, 0..1);
        let plan = req
            .plan(tt_isa::l1::DATA)
            .map_err(|e| invalid(&e.to_string()))?;
        let ready = plan.semaphore(ready);
        let done = plan.semaphore(done);
        let mut program = sync::take(ready, Before::EVERYTHING).to_vec();
        program.push(backend::set_thread_entry(thread::CFG_STATE_ID_StateID.addr32(), 0).unwrap());
        for r in [0, 8, 9, 10, 11, 12, 13, 14, 15] {
            program.extend(backend::set_gpr(r, 0).unwrap());
        }
        Ok(Self {
            plan,
            source,
            dest,
            bytes,
            ready,
            done,
            program,
        })
    }
    fn at(&self, b: Buf, offset: u64, length: u64) -> Result<u64> {
        if b != self.source && b != self.dest {
            return Err(invalid("foreign XMOV buffer"));
        }
        if offset.checked_add(length).is_none_or(|n| n > self.bytes) {
            return Err(invalid("XMOV transfer leaves declared buffer"));
        }
        self.plan
            .addr(b)
            .checked_add(offset)
            .ok_or_else(|| invalid("XMOV address overflow"))
    }
    fn scalar(&mut self, src: Option<u64>, dst: u64, width: TransferWidth) {
        // Rebase every scalar transfer: offsets are always 0..15, increment is
        // None, so neither a final update nor a long transfer can wrap a half.
        if let Some(src) = src {
            self.program
                .extend(backend::set_gpr(12, (src / 16) as u32).unwrap());
            self.program
                .extend(backend::set_gpr(14, (src % 16) as u32).unwrap());
            self.program.push(
                scalar::load_indirect(
                    width,
                    OffsetHalf::new(28).unwrap(),
                    OffsetIncrement::None,
                    8,
                    12,
                )
                .unwrap(),
            );
            self.program
                .push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
        } else {
            self.program.extend(backend::set_gpr(8, 0).unwrap());
        }
        self.program
            .extend(backend::set_gpr(13, (dst / 16) as u32).unwrap());
        self.program
            .extend(backend::set_gpr(15, (dst % 16) as u32).unwrap());
        self.program.push(
            scalar::store_indirect_l1(
                width,
                OffsetHalf::new(30).unwrap(),
                OffsetIncrement::None,
                8,
                13,
            )
            .unwrap(),
        );
        self.program
            .push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
    }
    fn transfer(&mut self, src: Option<(Buf, u64)>, dest: (Buf, u64), length: u64) -> Result<()> {
        let dst = self.at(dest.0, dest.1, length)?;
        let src = src.map(|(b, o)| self.at(b, o, length)).transpose()?;
        if let Some(s) = src {
            if length != 0 && s < dst + length && dst < s + length {
                return Err(invalid("XMOV copy ranges overlap"));
            }
        }
        if length == 0 {
            return Ok(());
        }
        let bulk = if dst % 16 == 0 && src.is_none_or(|s| s % 16 == 0) {
            length / 16 * 16
        } else {
            0
        };
        if bulk / 16 > 65535 {
            return Err(invalid("XMOV exceeds 65535 blocks"));
        }
        if bulk != 0 {
            // C9 precedes any configuration mutation; C12 precedes XMOV.
            self.program
                .push(backend::wait_for_mover(Before::EVERYTHING).unwrap());
            let mut cfg = ConfigWords::new();
            cfg.set(
                thcon::THCON_SEC0_REG6_Source_address,
                (src.unwrap_or(0) / 16) as u32,
            )
            .unwrap();
            cfg.set(
                thcon::THCON_SEC0_REG6_Destination_address,
                (dst / 16) as u32,
            )
            .unwrap();
            cfg.set(thcon::THCON_SEC0_REG6_Buffer_size, (bulk / 16) as u32)
                .unwrap();
            cfg.set(
                thcon::THCON_SEC0_REG6_Transfer_direction,
                if src.is_some() { 3 } else { 0 },
            )
            .unwrap();
            let mut words = vec![backend::nop(); cfg.program_len()];
            let n = cfg.program(0, &mut words).unwrap();
            self.program.extend(words.into_iter().take(n));
            self.program.push(
                backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY).unwrap(),
            );
            self.program.push(encode::xmov().unwrap());
            self.program
                .push(backend::wait_for_mover(Before::EVERYTHING).unwrap());
        }
        let mut offset = bulk;
        while offset < length {
            let remaining = length - offset;
            let width = [
                TransferWidth::Word,
                TransferWidth::Halfword,
                TransferWidth::Byte,
            ]
            .into_iter()
            .find(|w| {
                let n = u64::from(w.bytes());
                remaining >= n
                    && (dst + offset) % n == 0
                    && src.is_none_or(|s| (s + offset) % n == 0)
            })
            .unwrap();
            self.scalar(src.map(|s| s + offset), dst + offset, width);
            offset += u64::from(width.bytes());
        }
        Ok(())
    }
    fn finish(mut self) -> Arc<[Vec<Instruction>; 3]> {
        self.program
            .push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
        self.program
            .push(backend::wait_for_mover(Before::EVERYTHING).unwrap());
        self.program.push(sync::post(self.done));
        // The streaming T2 runner reserves output before this shell. Hand that
        // ownership to T0, then block publication until T0 drained both engines.
        let mut publish = vec![sync::post(self.ready)];
        publish.extend(sync::take(self.done, Before::EVERYTHING));
        Arc::new([self.program, vec![], publish])
    }
}
fn transfer(range: tt_isa::dram::DramRange, read: bool, l1: u64, bytes: u32) -> [u32; 8] {
    [
        if read { op::READ } else { op::WRITE },
        range.channel().index() as u32,
        0,
        (range.offset() + if read { 0 } else { TILE_DATA }) as u32,
        (l1 + if read { 0 } else { TILE_DATA }) as u32,
        bytes,
        0,
        0,
    ]
}
pub(crate) fn jobs(
    source: Option<&Placement>,
    output: &Placement,
    dims: [usize; 2],
    elem_bytes: u64,
) -> Result<Vec<Job>> {
    let count = geometry(dims, elem_bytes)?;
    if ![2, 4].contains(&elem_bytes)
        || output.tiles() != count
        || source.is_some_and(|s| s.tiles() != count)
    {
        return Err(invalid("XMOV storage/shape mismatch"));
    }
    let slot = 1024 * elem_bytes + 64;
    let mut jobs = Vec::with_capacity(count);
    for tile in 0..count {
        let mut local = Local::new(slot)?;
        let src = local.plan.addr(local.source);
        let dst = local.plan.addr(local.dest);
        let height = (dims[0] - tile / dims[1].div_ceil(32) * 32).min(32);
        let width = (dims[1] - tile % dims[1].div_ceil(32) * 32).min(32);
        if source.is_some() && height == 32 && width == 32 {
            local.transfer(
                Some((local.source, TILE_DATA)),
                (local.dest, TILE_DATA),
                1024 * elem_bytes,
            )?;
        } else {
            local.transfer(None, (local.dest, TILE_DATA), 1024 * elem_bytes)?;
            if source.is_some() {
                for face in 0..4u64 {
                    for row in 0..16u64 {
                        if face / 2 * 16 + row >= height as u64 || face % 2 * 16 >= width as u64 {
                            continue;
                        }
                        let offset = TILE_DATA + (face * 256 + row * 16) * elem_bytes;
                        let length = (width as u64 - face % 2 * 16).min(16) * elem_bytes;
                        local.transfer(
                            Some((local.source, offset)),
                            (local.dest, offset),
                            length,
                        )?;
                    }
                }
            }
        }
        let mut job = vec![];
        job.push(Step::List {
            what: "XMOV gather",
            entries: source.map_or_else(Vec::new, |source| {
                vec![transfer(source.slot(tile), true, src, slot as u32)]
            }),
        });
        let init = local.plan.semaphore_init();
        job.push(Step::Kernel {
            roles: local.finish(),
            init,
            mop: Box::new([None; 3]),
            loops: Default::default(),
            half: None,
        });
        job.push(Step::List {
            what: "XMOV scatter",
            entries: vec![transfer(
                output.slot(tile),
                false,
                dst,
                (1024 * elem_bytes) as u32,
            )],
        });
        jobs.push(job);
    }
    Ok(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declared_extents_overlap_overflow_and_noop() {
        let mut local = Local::new(4096).unwrap();
        let n = local.program.len();
        local
            .transfer(Some((local.source, 4096)), (local.dest, 4096), 0)
            .unwrap();
        assert_eq!(local.program.len(), n);
        for (src, dst, len) in [(0, 0, 4097), (4096, 0, 1), (u64::MAX, 0, 1)] {
            assert!(local
                .transfer(Some((local.source, src)), (local.dest, dst), len)
                .is_err());
            assert_eq!(local.program.len(), n);
        }
        assert!(local
            .transfer(Some((local.source, 0)), (local.source, 1), 16)
            .is_err());
        for dims in [[0, 1], [1, 0], [usize::MAX, 2], [usize::MAX, usize::MAX]] {
            assert!(geometry(dims, 4).is_err());
        }
    }
    #[test]
    fn natural_tails_and_program_ownership_audit() {
        let mut local = Local::new(4096).unwrap();
        local
            .transfer(Some((local.source, 0)), (local.dest, 0), 23)
            .unwrap();
        let roles = local.finish();
        assert!(roles[1].is_empty());
        assert!(roles[2]
            .iter()
            .all(|i| matches!(i.def().mnemonic(), "SEMPOST" | "SEMWAIT" | "SEMGET")));
        let names: Vec<_> = roles[0].iter().map(|i| i.def().mnemonic()).collect();
        assert_eq!(names.iter().filter(|&&n| n == "XMOV").count(), 1);
        assert_eq!(names.iter().filter(|&&n| n == "LOADIND").count(), 3);
        let sizes: Vec<_> = roles[0]
            .iter()
            .filter(|i| i.def().mnemonic() == "LOADIND")
            .map(|i| i.operand("Size").unwrap())
            .collect();
        assert_eq!(sizes, [1, 2, 3]);
    }
    // Byte-array interpreter of the memory/GPR pseudocode. This does not call
    // movement helpers or derive expected payloads from the builder's spans.
    fn model(program: &[Instruction], memory: &mut [u8]) {
        let mut g = [0xdeadbeefu32; 64];
        let mut cfg = [0u32; 224];
        for i in program {
            let field = |name| i.operand(name).unwrap() as usize;
            match i.def().mnemonic() {
                "SETDMAREG" => {
                    let h = field("ResultHalfReg");
                    let shift = h % 2 * 16;
                    g[h / 2] =
                        (g[h / 2] & !(65535 << shift)) | ((field("NewValue") as u32) << shift);
                }
                "WRCFG" => {
                    assert_eq!(field("Is128Bit"), 0);
                    cfg[field("CfgIndex")] = g[field("InputReg")];
                }
                "XMOV" => {
                    let (src, dst, n, mode) = (
                        cfg[88] as usize * 16,
                        cfg[89] as usize * 16,
                        (cfg[90] & 65535) as usize * 16,
                        cfg[90] >> 30,
                    );
                    assert!(n > 0);
                    match mode {
                        0 => memory[dst..dst + n].fill(0),
                        3 => memory.copy_within(src..src + n, dst),
                        _ => panic!("unrestricted mover mode"),
                    }
                }
                "LOADIND" | "STOREIND" => {
                    let load = i.def().mnemonic() == "LOADIND";
                    let half = field("OffsetHalfReg");
                    let addr = g[field("AddrReg")] as usize * 16
                        + ((g[half / 2] >> (half % 2 * 16)) & 65535) as usize;
                    assert_eq!(
                        field("OffsetIncrement"),
                        0,
                        "consumer rebases instead of wrapping offsets"
                    );
                    let n = if load {
                        [16, 4, 2, 1][field("Size")]
                    } else {
                        [16, 2, 4, 1][field("Size")]
                    };
                    assert_eq!(addr % n, 0);
                    let data = field(if load { "ResultReg" } else { "DataReg" });
                    for b in 0..n {
                        let r = data + b / 4;
                        let shift = b % 4 * 8;
                        if load {
                            g[r] = (g[r] & !(255 << shift)) | ((memory[addr + b] as u32) << shift);
                        } else {
                            memory[addr + b] = (g[r] >> shift) as u8;
                        }
                    }
                }
                "NOP" | "SETC16" | "STALLWAIT" | "SEMWAIT" | "SEMGET" | "SEMPOST" => {}
                other => panic!("unexpected movement instruction {other}"),
            }
        }
    }
    #[test]
    fn independent_byte_model_covers_logical_bits_zero_padding_and_guards() {
        use crate::tensor::{DramAlloc, DramTensor, Elem};
        for bytes in [2, 4] {
            for dims in [[1, 1], [1, 17], [17, 31], [32, 32], [33, 65], [97, 99]] {
                for copy in [false, true] {
                    let mut alloc = DramAlloc::new(&tt_isa::dram::Dram::FULL);
                    let input =
                        DramTensor::alloc_elem(&mut alloc, dims[0], dims[1], Elem::F32).unwrap();
                    let output =
                        DramTensor::alloc_elem(&mut alloc, dims[0], dims[1], Elem::F32).unwrap();
                    let work = jobs(
                        copy.then_some(&input.placement),
                        &output.placement,
                        dims,
                        bytes,
                    )
                    .unwrap();
                    for (tile, job) in work.iter().enumerate() {
                        let Step::Kernel {
                            roles, init, half, ..
                        } = &job[1]
                        else {
                            panic!("kernel")
                        };
                        assert_eq!(init.len(), 2);
                        assert_eq!(*half, None);
                        let Step::List { entries, .. } = job.last().unwrap() else {
                            panic!("scatter")
                        };
                        let dst = entries[0][4] as usize;
                        let slot = 1024 * bytes as usize + 64;
                        let local = Local::new(slot as u64).unwrap();
                        let src = local.plan.addr(local.source) as usize;
                        let mut memory = vec![0xa5; tt_isa::l1::DATA.end as usize];
                        let source: Vec<_> = (0..slot)
                            .map(|i| ((i * 37 + tile * 19) % 251) as u8)
                            .collect();
                        memory[src..src + slot].copy_from_slice(&source);
                        model(&roles[0], &mut memory);
                        let mut want = vec![0; 1024 * bytes as usize];
                        if copy {
                            for r in 0..32 {
                                for c in 0..32 {
                                    if tile / dims[1].div_ceil(32) * 32 + r < dims[0]
                                        && tile % dims[1].div_ceil(32) * 32 + c < dims[1]
                                    {
                                        let index =
                                            (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16;
                                        let off = index * bytes as usize;
                                        want[off..off + bytes as usize].copy_from_slice(
                                            &source[16 + off..16 + off + bytes as usize],
                                        );
                                    }
                                }
                            }
                        }
                        assert_eq!(&memory[dst..dst + want.len()], want);
                        assert_eq!(&memory[src..src + slot], source);
                        assert_eq!(&memory[dst - 16..dst], &[0xa5; 16]);
                        assert!(memory[dst + want.len()..dst + want.len() + 48]
                            .iter()
                            .all(|&b| b == 0xa5));
                    }
                }
            }
        }
    }
}
