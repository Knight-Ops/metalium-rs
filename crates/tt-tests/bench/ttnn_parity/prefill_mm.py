import sys, os, math, time, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--modes", default="bf16:bf16:HiFi2,bf16:bf16:HiFi4,bf16:bfp8:LoFi,bfp8:bfp8:LoFi")
ap.add_argument("--configs", default="auto,2d")
ap.add_argument("--out", default="prefill_mm.jsonl")
a = ap.parse_args()

SHAPES = [("2048x2048x16384", 2048, 2048, 16384), ("1024x8192x2048", 1024, 8192, 2048),
          ("2048x8192x2048", 2048, 8192, 2048), ("4096x4096x4096", 4096, 4096, 4096)]

def subblock(per_core_M, per_core_N, fp32):
    lim = 4 if fp32 else 8
    best = (1, 1)
    for h in range(1, per_core_M + 1):
        if per_core_M % h: continue
        for w in range(1, per_core_N + 1):
            if per_core_N % w: continue
            if h * w <= lim and h * w > best[0] * best[1]:
                if h == 1 or w == per_core_N:  # ttnn requires full-width rows when h>1
                    best = (h, w)
    return best

def cfg_2d(M, K, N, gx, gy, fp32, in0_block_w):
    pm = math.ceil(M / 32 / gy); pn = math.ceil(N / 32 / gx)
    sh, sw = subblock(pm, pn, fp32)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=in0_block_w, out_subblock_h=sh, out_subblock_w=sw,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=False)

out = Out(a.out)
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
grid = dev.compute_with_storage_grid_size()
out.emit({"kind": "conditions", "compute_grid": [grid.x, grid.y],
          "aiclk_sysfs": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
g = torch.Generator().manual_seed(2)
try:
    for name, M, K, N in SHAPES:
        if a.only and a.only not in name:
            continue
        x = torch.randn(1, 1, M, K, generator=g).bfloat16()
        w = (torch.randn(1, 1, K, N, generator=g) / math.sqrt(K)).bfloat16()
        ref = (x.float() @ w.float()).double()
        flops = 2.0 * M * N * K
        for mode in a.modes.split(","):
            adt, wdt, fid = mode.split(":")
            fp32 = (fid == "HiFi4")  # HiFi4 rows with fp32 dest acc; HiFi2/LoFi rows without (see notes)
            for fp32 in ((True, False) if fid != "HiFi4" else (True,)):
                tx = ttnn.from_torch(x, dtype=DT[adt], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                tw = ttnn.from_torch(w, dtype=DT[wdt], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                cfg = ckc(dev, fid, fp32)
                for conf in a.configs.split(","):
                    rec = {"kind": "prefill_matmul", "shape": name, "M": M, "K": K, "N": N, "act_dtype": adt,
                           "weight_dtype": wdt, "fidelity": fid, "fp32_dest_acc": fp32, "packer_l1_acc": True,
                           "config": conf, "mem": "DRAM interleaved in/out, bf16 out"}
                    try:
                        if conf == "auto":
                            op = lambda: ttnn.matmul(tx, tw, dtype=ttnn.bfloat16, compute_kernel_config=cfg,
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        else:
                            # 2D mcast on an 8x8 grid as tt_transformers' matmul_config (prefill rows 8):
                            # grid x over N, y over M; in0_block_w the largest divisor <= 8 of K tiles / 8 ... capped by L1
                            gx, gy = 8, 8
                            ibw = 4 if (fp32 or adt == "bf16") else 8
                            while (K // 32) % ibw: ibw //= 2
                            pc = cfg_2d(M, K, N, gx, gy, fp32, ibw)
                            rec.update({"grid": [gx, gy], "per_core_M": pc.per_core_M, "per_core_N": pc.per_core_N,
                                        "in0_block_w": pc.in0_block_w, "out_subblock": [pc.out_subblock_h, pc.out_subblock_w]})
                            op = lambda: ttnn.matmul(tx, tw, program_config=pc, dtype=ttnn.bfloat16,
                                                     compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        tid, res, o = per_op(dev, op, r_lo=1, r_hi=6, warm=3, samples=15)
                        got = ttnn.to_torch(o).double()
                        ttnn.release_trace(dev, tid); ttnn.deallocate(o)
                        rec.update(res)
                        rec["TFLOPS_slope"] = flops / (res["slope_us"] * 1e-6) / 1e12
                        rec["TFLOPS_trace1"] = flops / (res["trace1_us"]["median"] * 1e-6) / 1e12
                        rec["pcc"] = pcc(got, ref)
                    except Exception as e:
                        rec["error"] = str(e)[:600]
                        traceback.print_exc()
                    out.emit(rec)
                ttnn.deallocate(tx); ttnn.deallocate(tw)
finally:
    out.emit({"kind": "conditions_end", "aiclk_sysfs": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
    ttnn.close_device(dev)
