//! B copies selected raw datums using checked resident indices; no arithmetic.
use crate::tensor::{DramTensor, Elem, Job, Placement, Result, Step, TensorError, TransferBatch};
use tt_isa::dm::{op, TILE_DATA, TILE_SLOT};

pub(crate) fn jobs(
    input: &Placement,
    indices: &DramTensor,
    output: &Placement,
    [rows, cols]: [usize; 2],
    bytes: u32,
) -> Result<Vec<Job>> {
    indices.expect("resident gather indices", Elem::I32)?;
    if rows == 0
        || rows > 32
        || cols == 0
        || cols > i32::MAX as usize
        || [indices.rows, indices.cols] != [rows, 1]
    {
        return Err(TensorError::Shape(
            "resident gather needs a bounded row batch and one index per row".into(),
        ));
    }
    let mut req = crate::l1::Requirements::new(1);
    let idx = req.scratch("resident indices", TILE_SLOT, 64, 0..1);
    let src = req.scratch("indexed source tile", TILE_SLOT, 64, 0..1);
    let dst = req.scratch("indexed output tile", TILE_SLOT, 64, 0..1);
    let zero = req.scratch("indexed output initialization", TILE_SLOT, 64, 0..1);
    let layout = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (idx, src, dst, zero) = (
        layout.addr(idx),
        layout.addr(src),
        layout.addr(dst),
        layout.addr(zero),
    );
    let transfer = |range: tt_isa::dram::DramRange, at, read| {
        [
            if read { op::READ } else { op::WRITE },
            range.channel().index() as u32,
            0,
            range.offset() as u32,
            at as u32,
            range.len() as u32,
            0,
            0,
        ]
    };
    let mut batches = Vec::new();
    for tile in 0..cols.div_ceil(32) {
        let mut read = vec![transfer(indices.tile(0, 0), idx, true)];
        if tile == 0 {
            let half = bytes == 2;
            read.push([
                if half { op::FILL_HALFWORDS } else { op::FILL },
                0,
                1 | (1 << 8),
                zero as u32,
                0,
                0,
                0,
                0,
            ]);
            read.push([
                if half {
                    op::COPY_HALFWORDS
                } else {
                    op::COPY_WORDS
                },
                (zero + TILE_DATA + u64::from(bytes)) as u32,
                (dst + TILE_DATA) as u32,
                1024,
                0,
                bytes,
                0,
                0,
            ]);
        } else {
            read.push(transfer(output.slot(0), dst, true));
        }
        read.push(transfer(input.slot(tile), src, true));
        read.push([
            op::INDEX_PICK,
            idx as u32,
            src as u32,
            dst as u32,
            cols as u32,
            (tile * 32) as u32,
            rows as u32,
            bytes,
        ]);
        batches.push(TransferBatch {
            read,
            write: vec![transfer(output.slot(0), dst, false)],
        });
    }
    Ok(vec![vec![Step::Transfer {
        what: "resident indexed copies",
        depth: 1,
        batches,
    }]])
}
