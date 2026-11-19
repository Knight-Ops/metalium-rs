
# Block-float KV cache: keys and values stored BFP8 or BFP4

## The format

A block-float tile is a 16-byte header, 64 exponent bytes (one a face-row: 16 datums), then the datums
face-row by face-row (16 bytes a face-row at BFP8, 8 at BFP4). **A row of a tile is two face-rows and their two
exponents, independent of every other row**: a position's keys encoded alone are the bytes the whole cache's
encoding holds for that row. So a cache write can move encoded rows, never re-encoding a tile, and the cache's
bytes are the encoding of the F32 cache whatever order the rows came in.

## Not done

Flash attention: BFP2 per-layer formats
