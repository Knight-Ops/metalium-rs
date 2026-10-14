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
    select_taps_jobs(table, indices, table.rows, output, units)
}

/// The columns of a [`select_taps_jobs`] output for `taps` taps of `table`:
/// every tap but the last a whole number of tile columns, so each tap's
/// first column is a tile's.
pub fn select_taps_cols(table_cols: usize, taps: usize) -> usize {
    (taps - 1) * table_cols.div_ceil(32) * 32 + table_cols
}

/// [`select_rows_jobs`] for several index rows at once: `indices` is a
/// `[taps, n]` I32 tensor and `output`, `[n, select_taps_cols(table.cols,
/// taps)]`, holds at row `i` the rows `indices[t][i]` of `table` side by side,
/// tap `t` from tile column `t * ct` (`ct` the table's tile columns) -- an
/// im2col's patch matrix when the table is an NHWC activation and the
/// indices its windows. A tap's columns past `table.cols` are the table's
/// padding, as the last tap's are.
///
/// `bound` is the number of rows an index may name: `table.rows`, or one more
/// when the table's first padding row is in its last tile row (`table.rows %
/// 32 != 0`) and the caller has made it zero, so a padding tap reads zeros.
///
/// Tap `t`'s index is row `t % 32` of its index tile: the record reads its 32
/// indices from row 0 of the slot at `idx_at`, so `idx_at` is moved by that
/// row's place in the face layout. A batch is one output tile row and up to
/// [`SELECT_TILES`] of its tiles; it reads the (at most two) index tiles its
/// taps are in, then one record a tap.
pub(crate) fn select_taps_jobs(
    table: &DramTensor,
    indices: &DramTensor,
    bound: usize,
    output: &DramTensor,
    units: usize,
) -> Result<Vec<Job>> {
    indices.expect("resident row select indices", Elem::I32)?;
    let (taps, n) = (indices.rows, indices.cols);
    let [trt, ct] = table.grid();
    if n == 0
        || taps == 0
        || table.rows == 0
        || table.cols == 0
        || bound < table.rows
        || bound > trt * 32
        || bound > i32::MAX as usize
        || [output.rows, output.cols] != [n, select_taps_cols(table.cols, taps)]
        || output.elem != table.elem
    {
        return Err(TensorError::Shape(
            "a resident row select needs [taps, n] indices, a bound within the table's tile \
             rows and an [n, taps x cols] output of the table's"
                .into(),
        ));
    }
    let mut req = crate::l1::Requirements::new(1);
    let idx = req.scratch("row select index tiles", 2 * TILE_SLOT, 64, 0..1);
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
    if !tt_isa::l1::DATA.contains(idx, 2 * TILE_SLOT)
        || !tt_isa::l1::DATA.contains(stage, 2 * SELECT_TILES as u64 * TILE_SLOT)
    {
        return Err(TensorError::Shape(
            "the row select's slots are outside the data arena".into(),
        ));
    }
    let ort = output.grid()[0];
    // One batch a tile row of the output and a group of its tap-major tiles.
    let tiles = taps * ct;
    let groups = tiles.div_ceil(SELECT_TILES);
    let table_ref = table.tensor_ref().encode();
    let mut jobs = Vec::new();
    for run in runs(ort * groups, units, SELECT_BATCHES) {
        let batches = run
            .enumerate()
            .map(|(b, item)| {
                let (oi, g) = (item / groups, item % groups);
                let (k0, k1) = (g * SELECT_TILES, ((g + 1) * SELECT_TILES).min(tiles));
                let count = 32.min(n - oi * 32);
                let first = stage + (b % 2 * SELECT_TILES) as u64 * TILE_SLOT;
                // The index tile rows this group's taps are in: at most two,
                // as a group holds at most `SELECT_TILES` < 32 taps.
                let tile_rows = [k0 / ct / 32, (k1 - 1) / ct / 32];
                let mut read = Vec::new();
                for (s, &tr) in tile_rows.iter().enumerate() {
                    if s == 1 && tr == tile_rows[0] {
                        break;
                    }
                    let range = indices.tile(tr, oi);
                    read.push([
                        op::READ,
                        range.channel().index() as u32,
                        0,
                        range.offset() as u32,
                        (idx + s as u64 * TILE_SLOT) as u32,
                        range.len() as u32,
                        0,
                        0,
                    ]);
                }
                if count < 32 {
                    for k in 0..k1 - k0 {
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
                // One record a tap: its tile columns in this group.
                let mut k = k0;
                while k < k1 {
                    let (t, j0) = (k / ct, k % ct);
                    let j1 = ct.min(j0 + (k1 - k));
                    let slot = idx + if t / 32 == tile_rows[0] { 0 } else { TILE_SLOT };
                    let q = (t % 32) as u64;
                    let row = (q / 16 * 512 + q % 16 * 16) * 4;
                    read.push([
                        record::SELECT_ROWS,
                        (slot + row) as u32,
                        count as u32,
                        bound as u32,
                        j0 as u32,
                        (j1 - j0) as u32,
                        (first + (k - k0) as u64 * TILE_SLOT) as u32,
                        0,
                    ]);
                    read.extend(table_ref);
                    k += j1 - j0;
                }
                let write = (k0..k1)
                    .map(|k| {
                        let to = output.tile(oi, k);
                        let slot = first + (k - k0) as u64 * TILE_SLOT;
                        [
                            op::WRITE,
                            to.channel().index() as u32,
                            (k - k0) as u32 % tt_isa::dram::PORTS as u32,
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

/// Tile row `tile_row` of `out` all zeros, padding too: one staged tile made
/// zero on the card (a `FILL` leaves datum `(0, 0)` and zeroes the rest; a
/// `COPY_WORDS` spreads datum `(0, 1)` over all of them) and written to each
/// tile of the row. For [`crate::session::Session::select_taps`]'s zero row,
/// where a table with whole tile rows is copied into one a row longer.
pub(crate) fn zero_tile_row_jobs(out: &DramTensor, tile_row: usize) -> Result<Vec<Job>> {
    let [rt, ct] = out.grid();
    if tile_row >= rt {
        return Err(TensorError::Shape(format!(
            "tile row {tile_row} of a tensor of {rt}"
        )));
    }
    let mut req = crate::l1::Requirements::new(1);
    let zero = req.scratch("zero tile", TILE_SLOT, 64, 0..1);
    let layout = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let zero = layout.addr(zero);
    let batches = (0..ct)
        .collect::<Vec<_>>()
        .chunks(128)
        .map(|js| TransferBatch {
            read: vec![
                [op::FILL, 0, fill::param(1, 1), zero as u32, 0, 0, 0, 0],
                [
                    op::COPY_WORDS,
                    (zero + TILE_DATA + 4) as u32,
                    (zero + TILE_DATA) as u32,
                    1024,
                    0,
                    4,
                    0,
                    0,
                ],
            ],
            write: js
                .iter()
                .enumerate()
                .map(|(k, &j)| {
                    let to = out.tile(tile_row, j);
                    [
                        op::WRITE,
                        to.channel().index() as u32,
                        k as u32 % tt_isa::dram::PORTS as u32,
                        (to.offset() + TILE_DATA) as u32,
                        (zero + TILE_DATA) as u32,
                        4096,
                        0,
                        0,
                    ]
                })
                .collect(),
        })
        .collect();
    Ok(vec![vec![Step::Transfer {
        what: "zero tile row",
        depth: 1,
        batches,
    }]])
}
