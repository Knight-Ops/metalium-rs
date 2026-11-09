import sys, os, math, time, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--modes", default="bf16:HiFi2,bf16:HiFi4,bfp8:HiFi2,bfp8:LoFi,bfp4:LoFi")
ap.add_argument("--out", default="decode_mm.jsonl")
ap.add_argument("--plain", action="store_true", help="also time plain ttnn.linear (DRAM interleaved, auto config)")
a = ap.parse_args()

SHAPES = [  # name, M, K, N
    ("qkv", 32, 2048, 6144),
    ("o", 32, 2048, 2048),
    ("gate_up_fused", 32, 2048, 16384),
    ("gate_or_up", 32, 2048, 8192),
    ("down", 32, 8192, 2048),
    ("head", 32, 2048, 49152),
]

def find_grid_k_n(K, N):  # tt_transformers model_config.find_grid_k_n
    max_rows, max_cols = 8, 8
    cands = sorted([c for c in range(1, 65) if K % c == 0 and N % c == 0], reverse=True)
    for cores in cands:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and cores // rows <= max_cols:
                return rows, cores // rows
    raise AssertionError

def largest_div(n, m=8):
    for i in range(m, 0, -1):
        if n % i == 0:
            return i
    return 1

out = Out(a.out)
dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=0)
dram = dev.dram_grid_size()
grid = dev.compute_with_storage_grid_size()
out.emit({"kind": "conditions", "dram_grid": [dram.x, dram.y], "compute_grid": [grid.x, grid.y],
          "aiclk_sysfs_before": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
dram_cores = dram.x
dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
g = torch.Generator().manual_seed(1)
try:
    for name, M, K, N in SHAPES:
        if a.only and a.only not in name:
            continue
        x = torch.randn(1, 1, M, K, generator=g).bfloat16()
        w = (torch.randn(1, 1, K, N, generator=g) / math.sqrt(K)).bfloat16()
        ref = (x.double() @ w.double())
        rows, cols = find_grid_k_n(K // 32, N // 32)
        ncores = rows * cols
        core_grid = ttnn.CoreGrid(y=rows, x=cols)
        in0_mem = ttnn.create_sharded_memory_config((M, K // ncores), core_grid, ttnn.ShardStrategy.WIDTH,
                                                    ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        out_mem = ttnn.create_sharded_memory_config((M, N // ncores), core_grid, ttnn.ShardStrategy.WIDTH,
                                                    ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        padded = math.ceil(N / (32 * dram_cores)) * 32 * dram_cores
        w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                                  ttnn.ShardSpec(dram_grid, (K, padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR))
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=largest_div(K // (32 * ncores)), per_core_M=math.ceil(M / 32),
            per_core_N=math.ceil(N / (32 * ncores)), fused_activation=None)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=in0_mem)
        for mode in a.modes.split(","):
            wdt, fid = mode.split(":")
            for fp32 in (True,):
                rec = {"kind": "decode_matmul", "shape": name, "M": M, "K": K, "N": N, "weight_dtype": wdt,
                       "fidelity": fid, "fp32_dest_acc": fp32, "packer_l1_acc": True, "math_approx": False,
                       "config": "DRAMSharded", "grid": [cols, rows], "ncores": ncores,
                       "in0_block_w": pc.in0_block_w, "per_core_N": pc.per_core_N,
                       "act": "bf16 L1 width-sharded", "out": "bf16 L1 width-sharded",
                       "weight_mem": f"DRAM width-sharded over {dram_cores} banks, shard (K={K}, {padded // dram_cores})"}
                try:
                    tw = ttnn.from_torch(w, dtype=DT[wdt], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=w_mem)
                    cfg = ckc(dev, fid, fp32)
                    op = lambda: ttnn.matmul(tx, tw, program_config=pc, memory_config=out_mem,
                                             dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                    tid, res, o = per_op(dev, op)
                    got = ttnn.to_torch(o).double()
                    ttnn.release_trace(dev, tid); ttnn.deallocate(o)
                    wbytes = K * N * WT_BYTES[wdt]
                    rec.update(res)
                    rec["weight_bytes"] = wbytes
                    rec["GBps_slope"] = wbytes / (res["slope_us"] * 1e-6) / 1e9
                    rec["GBps_trace1"] = wbytes / (res["trace1_us"]["median"] * 1e-6) / 1e9
                    rec["pcc"] = pcc(got, ref)
                    rec["max_abs_err"] = float((got - ref).abs().max())
                    ttnn.deallocate(tw)
                except Exception as e:
                    rec["error"] = str(e)[:600]
                    traceback.print_exc()
                out.emit(rec)
        ttnn.deallocate(tx)
        if a.plain:
            # Plain ttnn.linear: activation and weight DRAM-interleaved, auto program config.
            txd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for mode in a.modes.split(","):
                wdt, fid = mode.split(":")
                rec = {"kind": "decode_matmul", "shape": name, "M": M, "K": K, "N": N, "weight_dtype": wdt,
                       "fidelity": fid, "fp32_dest_acc": True, "packer_l1_acc": True, "config": "plain ttnn.linear auto",
                       "act": "bf16 DRAM interleaved", "out": "bf16 DRAM interleaved"}
                try:
                    tw = ttnn.from_torch(w, dtype=DT[wdt], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    cfg = ckc(dev, fid, True)
                    op = lambda: ttnn.linear(txd, tw, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                    tid, res, o = per_op(dev, op)
                    got = ttnn.to_torch(o).double()
                    ttnn.release_trace(dev, tid); ttnn.deallocate(o)
                    wbytes = K * N * WT_BYTES[wdt]
                    rec.update(res)
                    rec["weight_bytes"] = wbytes
                    rec["GBps_slope"] = wbytes / (res["slope_us"] * 1e-6) / 1e9
                    rec["pcc"] = pcc(got, ref)
                    ttnn.deallocate(tw)
                except Exception as e:
                    rec["error"] = str(e)[:600]
                out.emit(rec)
            ttnn.deallocate(txd)
finally:
    out.emit({"kind": "conditions_end", "aiclk_sysfs_after": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
    ttnn.close_device(dev)
