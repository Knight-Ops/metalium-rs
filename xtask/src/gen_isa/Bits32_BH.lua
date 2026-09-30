-- Blackhole instruction layouts MEASURED against ttsim, where the pinned
-- specification draws only Wormhole's.
--
-- Same dialect as `Diagrams/Src/Bits32.lua`, read by the same parser, so an entry
-- here is held to the same grammar and structural checks as the specification's
-- own. It is not part of the specification and is not covered by its digest.
--
-- Every entry is `<KEY>_BH`, supersedes the Wormhole diagram `<KEY>`, and has a
-- row in `measured.rs` naming:
--   * the fields whose position was measured to differ from Wormhole's, and
--   * the gate that measured them.
-- Every other field is carried over from the Wormhole diagram unchanged -- and
-- `gen-isa` refuses the entry if one is not -- so carried fields remain exactly as
-- unverified as they were. `gen-isa` also refuses an entry once the specification
-- documents the instruction for Blackhole itself: at that point the measurement
-- is a cross-check against the document, not a substitute for it.

local diagrams = {
  MVMUL_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {14, 2, "AddrMod"},
      {19, 1, "BroadcastSrcBRow"},
      {22, 1, "FlipSrcA"},
      {23, 1, "FlipSrcB"},
      {24, 8, "0x26"},
    }
  end,
  MOVA2D_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {13, 1, "Move8Rows"},
      {14, 2, "AddrMod"},
      {17, 6, "SrcRow"},
      {23, 1, "UseDst32bLo"},
      {24, 8, "0x12"},
    }
  end,
  MOVB2D_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {11, 1, "BroadcastCol0"},
      {12, 1, "Broadcast1RowTo8"},
      {13, 1, "Move4Rows"},
      {14, 2, "AddrMod"},
      {17, 6, "SrcRow"},
      {23, 1, "UseDst32bLo"},
      {24, 8, "0x13"},
    }
  end,
}
