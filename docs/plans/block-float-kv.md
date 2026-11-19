# Block-float KV cache: keys and values stored BFP8 or BFP4

## The format

A block-float tile is a 16-byte header, 64 exponent bytes (one a face-row: 16 datums), then the datums
face-row by face-row (16 bytes a face-row at BFP8, 8 at BFP4). **A row of a tile is two face-rows and their two
exponents, independent of every other row**: a position's keys encoded alone are the bytes the whole cache's
encoding holds for that row. So a cache write can move encoded rows, never re-encoding a tile, and the cache's
bytes are the encoding of the F32 cache whatever order the rows came in.

## In the models

`WeightOptions::kv_block_float` (Qwen3.5): the cache is made interleaved (`with_row_storage`: a stored weight is
laid by column, which a row scatter does not take), its keys and values computed F32, written as encoded rows and
read where they lie; a trace holds it as its state (block-float state is written in place and copied back slot for
slot). Gate `qwen35_block_float_kv` (card 0): traced equals the eager fixed schedule bit for bit for BFP8 and BFP4,
and the logits differ from the BF16 cache's.

## Not done

Flash attention, BFP2, per-layer formats.
