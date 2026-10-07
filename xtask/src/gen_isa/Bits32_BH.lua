-- Blackhole instruction layouts MEASURED on ttsim or silicon, where the pinned
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
  UNPACR_NOP_SETDVALID_BH = function()
    return Bits32{
      {0, 9, "0x1e9"},
      {23, 1, "WhichUnpacker"},
      {24, 8, "0x43"},
    }
  end,
  UNPACR_NOP_ZEROSRC_BH = function()
    return Bits32{
      {0, 2, "1"},
      {2, 2, "NegativeInfSrcA"},
      {4, 1, "BothBanks"},
      {5, 1, "WaitLikeUnpacr"},
      {6, 1, "0"},
      {23, 1, "WhichUnpacker"},
      {24, 8, "0x43"},
    }
  end,
  GMPOOL_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {14, 1, "ArgMax"},
      {15, 2, "AddrMod"},
      {19, 1, "1"},
      {22, 1, "FlipSrcA"},
      {23, 1, "FlipSrcB"},
      {24, 8, "0x33"},
    }
  end,
  GAPOOL_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {15, 2, "AddrMod"},
      {19, 1, "1"},
      {22, 1, "FlipSrcA"},
      {23, 1, "FlipSrcB"},
      {24, 8, "0x34"},
    }
  end,
  MVMUL_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {14, 3, "AddrMod"},
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
      {14, 3, "AddrMod"},
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
      {14, 3, "AddrMod"},
      {17, 6, "SrcRow"},
      {23, 1, "UseDst32bLo"},
      {24, 8, "0x13"},
    }
  end,
  MOVD2A_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {13, 1, "Move4Rows"},
      {14, 3, "AddrMod"},
      {17, 6, "SrcRow"},
      {23, 1, "UseDst32bLo"},
      {24, 8, "0x08"},
    }
  end,
  MOVD2B_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {13, 1, "Move4Rows"},
      {14, 3, "AddrMod"},
      {17, 6, "SrcRow"},
      {23, 1, "UseDst32bLo"},
      {24, 8, "0x0A"},
    }
  end,
  MOVB2A_BH = function()
    return Bits32{
      {0, 6, "SrcBRow"},
      {13, 1, "Move4Rows"},
      {14, 3, "AddrMod"},
      {17, 6, "SrcARow"},
      {24, 8, "0x0B"},
    }
  end,
  ELWADD_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {14, 3, "AddrMod"},
      {19, 1, "BroadcastSrcBCol0"},
      {20, 1, "BroadcastSrcBRow"},
      {21, 1, "AddDst"},
      {22, 1, "FlipSrcA"},
      {23, 1, "FlipSrcB"},
      {24, 8, "0x28"},
    }
  end,
  ELWSUB_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {14, 3, "AddrMod"},
      {19, 1, "BroadcastSrcBCol0"},
      {20, 1, "BroadcastSrcBRow"},
      {21, 1, "AddDst"},
      {22, 1, "FlipSrcA"},
      {23, 1, "FlipSrcB"},
      {24, 8, "0x30"},
    }
  end,
  ELWMUL_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {14, 3, "AddrMod"},
      {19, 1, "BroadcastSrcBCol0"},
      {20, 1, "BroadcastSrcBRow"},
      {22, 1, "FlipSrcA"},
      {23, 1, "FlipSrcB"},
      {24, 8, "0x27"},
    }
  end,
  DOTPV_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {14, 3, "AddrMod"},
      {22, 1, "FlipSrcA"},
      {23, 1, "FlipSrcB"},
      {24, 8, "0x29"},
    }
  end,
  MOVDBGA2D_BH = function()
    return Bits32{
      {0, 10, "DstRow"},
      {13, 1, "Move8Rows"},
      {14, 3, "AddrMod"},
      {17, 6, "SrcRow"},
      {23, 1, "UseDst32bLo"},
      {24, 8, "0x09"},
    }
  end,
  SHIFTXB_BH = function()
    return Bits32{
      {0, 6, "SrcRow"},
      {10, 1, "ShiftInZero"},
      {14, 3, "AddrMod"},
      {24, 8, "0x18"},
    }
  end,
  ZEROACC_BH = function()
    return Bits32{
      {0, 10, "Imm10"},
      {14, 3, "AddrMod"},
      {18, 1, "UseDst32b"},
      {19, 2, "Mode"},
      {24, 8, "0x10"},
    }
  end,
}
