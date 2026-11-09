import sys, os, math, time, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--modes", default="bf16:HiFi2,bf16:HiFi4,bfp8:HiFi2,bfp8:LoFi,bfp4:LoFi")
ap.add_argument("--grids", default="8x1,8x2,8x4")
ap.add_argument("--out", default="decode_sweep.jsonl")
a = ap.parse_args()
SHAPES = [("qkv", 32, 2048, 6144, 1), ("o", 32, 2048, 2048, 1), ("gate_up_fused", 32, 2048, 16384, 1),
          ("gate_or_up", 32, 2048, 8192, 1), ("down", 32, 8192, 2048, 1),
          ("head_split6x8192", 32, 2048, 49152, 6)]
out = Out(a.out)
dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20)
dram = dev.dram_grid_size(); nb = dram.x
out.emit({"kind": "conditions", "dram_grid": [dram.x, dram.y], "aiclk_sysfs": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
g = torch.Generator().manual_seed(1)
try:
  for name, M, K, N, splits in SHAPES:
    if a.only and a.only not in name: continue
    Ns = N // splits
    x = torch.randn(1, 1, M, K, generator=g).bfloat16()
    ws = [(torch.randn(1, 1, K, Ns, generator=g) / math.sqrt(K)).bfloat16() for _ in range(splits)]
    refs = [x.double() @ w.double() for w in ws]
    padded = math.ceil(Ns / (32 * nb)) * 32 * nb
    w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                              ttnn.ShardSpec(dram_grid, (K, padded // nb), ttnn.ShardOrientation.ROW_MAJOR))
    for gs in a.grids.split(","):
        gx, gy = map(int, gs.split("x")); nc = gx * gy
        if (K // 32) % nc or (Ns // 32) % nc: continue
        kt = K // 32 // nc
        core_grid = ttnn.CoreGrid(y=gy, x=gx)
        in0_mem = ttnn.create_sharded_memory_config((M, K // nc), core_grid, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        out_mem = ttnn.create_sharded_memory_config((M, Ns // nc), core_grid, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=in0_mem)
        ibws = sorted({d for d in (kt, kt // 2, kt // 4) if d >= 1 and kt % d == 0}, reverse=True)
        for mode in a.modes.split(","):
            wdt, fid = mode.split(":")
            tws = [ttnn.from_torch(w, dtype=DT[wdt], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=w_mem) for w in ws]
            cfg = ckc(dev, fid, True)
            for ibw in ibws:
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=ibw, per_core_M=1, per_core_N=Ns // 32 // nc, fused_activation=None)
                rec = {"kind": "decode_matmul", "shape": name, "M": M, "K": K, "N": N, "splits": splits, "weight_dtype": wdt,
                       "fidelity": fid, "fp32_dest_acc": True, "packer_l1_acc": True, "config": "DRAMSharded", "in0_grid": [gx, gy],
                       "in0_block_w": ibw, "per_core_N": Ns // 32 // nc}
                try:
                    op = (lambda: ttnn.matmul(tx, tws[0], program_config=pc, memory_config=out_mem, dtype=ttnn.bfloat16, compute_kernel_config=cfg)) if splits == 1 else \
                         (lambda: [ttnn.matmul(tx, tw, program_config=pc, memory_config=out_mem, dtype=ttnn.bfloat16, compute_kernel_config=cfg) for tw in tws])
                    tid, res, o = per_op(dev, op)
                    os_ = o if isinstance(o, list) else [o]
                    p = min(pcc(ttnn.to_torch(oo).double(), r) for oo, r in zip(os_, refs))
                    ttnn.release_trace(dev, tid)
                    for oo in os_: ttnn.deallocate(oo)
                    wbytes = K * N * WT_BYTES[wdt]
                    rec.update(res); rec["weight_bytes"] = wbytes
                    rec["GBps_slope"] = wbytes / (res["slope_us"] * 1e-6) / 1e9
                    rec["pcc_min"] = p
                except Exception as e:
                    rec["error"] = str(e)[:400]
                out.emit(rec)
            for tw in tws: ttnn.deallocate(tw)
        ttnn.deallocate(tx)
finally:
    out.emit({"kind": "conditions_end", "aiclk_sysfs": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
    ttnn.close_device(dev)
