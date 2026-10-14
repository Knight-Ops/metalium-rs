//! B copies selected raw datums using checked resident indices; no arithmetic.
use crate::tensor::{
    runs, DramTensor, Elem, Job, Placement, Result, Step, TensorError, TransferBatch,
};
use tt_isa::dm::{fill, op, record, TILE_DATA, TILE_SLOT};

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

/// Output tile columns a [`select_rows_jobs`] batch stages: each half of the
/// staging holds this many tiles (about 100 KB), and the two halves alternate
/// so NC writes one out while B reads the next.
const SELECT_TILES: usize = 24;

/// Batches a job of [`select_rows_jobs`] may hold before the work is split
/// over more tiles than there are.
const SELECT_BATCHES: usize = 64;

/// A new `[n, table.cols]` tensor whose row `i` is row `indices[i]` of
/// `table`, bit for bit, `indices` an `[1, n]` I32 tensor on the card -- an
/// embedding's lookup, with no address built on the host: each batch reads
/// one tile of 32 indices and B's [`record::SELECT_ROWS`] turns them into
/// the table's face-row reads, so a lookup is `2 x` tile columns reads a
/// row, however many rows the table has. The job list depends on the shapes
/// alone (the index is read where it runs), so a trace replays it with the
/// indices it finds. An index outside `0..table.rows` stops the mover with
/// `DOMAIN` before it reads a datum.
///
/// `output`'s rows past `n` in its last tile row are zero (the batch's
/// `FILL`), its columns past `cols` are the table's.
pub(crate) fn select_rows_jobs(
    table: &DramTensor,
    indices: &DramTensor,
    output: &DramTensor,
    units: usize,
) -> Result<Vec<Job>> {
    indices.expect("resident row select indices", Elem::I32)?;
    let n = output.rows;
    if n == 0
        || table.rows == 0
        || table.cols == 0
        || table.rows > i32::MAX as usize
        || [indices.rows, indices.cols] != [1, n]
        || [output.rows, output.cols] != [n, table.cols]
        || output.elem != table.elem
    {
        return Err(TensorError::Shape(
            "a resident row select needs a [1, n] index and an [n, cols] output of the table's"
                .into(),
        ));
    }
    let mut req = crate::l1::Requirements::new(1);
    let idx = req.scratch("row select index tile", TILE_SLOT, 64, 0..1);
    let slots = req.scratch(
        "row select staged tiles",
        2 * SELECT_TILES as u64 * TILE_SLOT,
        64,
        0..1,
    );
    let layout = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (idx, stage) = (layout.addr(idx), layout.addr(slots));
    // The mover does not check that `SELECT_ROWS`'s slots are in the data
    // arena (`record::SELECT_ROWS`); they come from its plan, and this is the
    // check that says so.
    if !tt_isa::l1::DATA.contains(idx, TILE_SLOT)
        || !tt_isa::l1::DATA.contains(stage, 2 * SELECT_TILES as u64 * TILE_SLOT)
    {
        return Err(TensorError::Shape(
            "the row select's slots are outside the data arena".into(),
        ));
    }
    let [ort, oct] = output.grid();
    // One batch a tile row of the output and a group of its tile columns.
    let groups = oct.div_ceil(SELECT_TILES);
    let table_ref = table.tensor_ref().encode();
    let mut jobs = Vec::new();
    for run in runs(ort * groups, units, SELECT_BATCHES) {
        let batches = run
            .enumerate()
            .map(|(b, item)| {
                let (oi, g) = (item / groups, item % groups);
                let (j0, j1) = (g * SELECT_TILES, ((g + 1) * SELECT_TILES).min(oct));
                let count = 32.min(n - oi * 32);
                let first = stage + (b % 2 * SELECT_TILES) as u64 * TILE_SLOT;
                let range = indices.tile(0, oi);
                let mut read = vec![[
                    op::READ,
                    range.channel().index() as u32,
                    0,
                    range.offset() as u32,
                    idx as u32,
                    range.len() as u32,
                    0,
                    0,
                ]];
                if count < 32 {
                    for k in 0..j1 - j0 {
                        let at = first + k as u64 * TILE_SLOT;
                        read.push([
                            op::FILL,
                            0,
                            fill::param(count as u32, 32),
                            at as u32,
                            0,
                            0,
                            0,
                            0,
                        ]);
                    }
                }
                read.push([
                    record::SELECT_ROWS,
                    idx as u32,
                    count as u32,
                    table.rows as u32,
                    j0 as u32,
                    (j1 - j0) as u32,
                    first as u32,
                    0,
                ]);
                read.extend(table_ref);
                let write = (j0..j1)
                    .map(|oj| {
                        let to = output.tile(oi, oj);
                        let slot = first + (oj - j0) as u64 * TILE_SLOT;
                        [
                            op::WRITE,
                            to.channel().index() as u32,
                            (oj - j0) as u32 % tt_isa::dram::PORTS as u32,
                            (to.offset() + TILE_DATA) as u32,
                            (slot + TILE_DATA) as u32,
                            4096,
                            0,
                            0,
                        ]
                    })
                    .collect();
                TransferBatch { read, write }
            })
            .collect();
        jobs.push(vec![Step::Transfer {
            what: "resident row select",
            depth: 2,
            batches,
        }]);
    }
    Ok(jobs)
}
