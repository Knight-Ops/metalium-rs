"""2D-mcast sweep with out_block_h/out_block_w so large per-core outputs fit L1."""
import sys, os, math, time, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--modes", default="bf16:bf16:HiFi2:1,bf16:bf16:HiFi4:1,bf16:bfp8:LoFi:1,bfp8:bfp8:LoFi:0")
ap.add_argument("--grids", default="8x8,11x10")
ap.add_argument("--out", default="prefill_2d.jsonl")
a = ap.parse_args()
SHAPES = [("2048x2048x16384", 2048, 2048, 16384), ("1024x8192x2048", 1024, 8192, 2048),
          ("2048x8192x2048", 2048, 8192, 2048), ("4096x4096x4096", 4096, 4096, 4096)]
def divisors(n): return [d for d in range(1, n + 1) if n % d == 0]
def subblock(bh, bw, fp32):
    lim = 4 if fp32 else 8
    best = (1, 1)
    for h in divisors(bh):
        for w in divisors(bw):
            if h * w <= lim and h * w > best[0] * best[1] and (h == 1 or w == bw):
                best = (h, w)
    return best
out = Out(a.out)
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
out.emit({"kind": "conditions", "aiclk_sysfs": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
g = torch.Generator().manual_seed(2)
try:
  for name, M, K, N in SHAPES:
    if a.only and a.only not in name: continue
    x = torch.randn(1, 1, M, K, generator=g).bfloat16()
    w = (torch.randn(1, 1, K, N, generator=g) / math.sqrt(K)).bfloat16()
    ref = (x.float() @ w.float()).double(); flops = 2.0 * M * N * K
    for mode in a.modes.split(","):
        adt, wdt, fid, f32 = mode.split(":"); fp32 = f32 == "1"
        tx = ttnn.from_torch(x, dtype=DT[adt], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tw = ttnn.from_torch(w, dtype=DT[wdt], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cfg = ckc(dev, fid, fp32)
        best = None
        for gs in a.grids.split(","):
            gx, gy = map(int, gs.split("x"))
            pm = math.ceil(M / 32 / gy); pn = math.ceil(N / 32 / gx)
            obhs = sorted({d for d in divisors(pm) if d <= 8} | ({pm} if pm <= 8 else set()))[-1:]
            obws = sorted({d for d in divisors(pn) if d <= 16})[-2:]
            for obh in obhs:
                for obw in obws:
                    for ibw in (2, 4):
                        if (K // 32) % ibw: continue
                        sh, sw = subblock(obh, obw, fp32)
                        rec = {"kind": "prefill_matmul", "shape": name, "M": M, "K": K, "N": N, "act_dtype": adt, "weight_dtype": wdt,
                               "fidelity": fid, "fp32_dest_acc": fp32, "packer_l1_acc": True, "config": "2d-outblock",
                               "grid": [gx, gy], "per_core_M": pm, "per_core_N": pn, "out_block": [obh, obw], "in0_block_w": ibw,
                               "out_subblock": [sh, sw], "mem": "DRAM interleaved in/out, bf16 out"}
                        try:
                            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
                                out_block_h=obh, out_block_w=obw, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                                fused_activation=None, fuse_batch=False)
                            op = lambda: ttnn.matmul(tx, tw, program_config=pc, dtype=ttnn.bfloat16, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            tid, res, o = per_op(dev, op, r_lo=1, r_hi=6, warm=3, samples=15)
                            got = ttnn.to_torch(o).double(); ttnn.release_trace(dev, tid); ttnn.deallocate(o)
                            rec.update(res)
                            rec["TFLOPS_slope"] = flops / (res["slope_us"] * 1e-6) / 1e12
                            rec["TFLOPS_trace1"] = flops / (res["trace1_us"]["median"] * 1e-6) / 1e12
                            rec["pcc"] = pcc(got, ref)
                        except Exception as e:
                            rec["error"] = str(e)[:300]
                        out.emit(rec)
        ttnn.deallocate(tx); ttnn.deallocate(tw)
finally:
    out.emit({"kind": "conditions_end", "aiclk_sysfs": open("os.environ.get("TT_AICLK_SYSFS", "/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")").read().strip()})
    ttnn.close_device(dev)
