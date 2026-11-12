import json, math, statistics, time, sys, os
import torch, ttnn

def stats(xs):
    s = sorted(xs)
    pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]
    return {"median": statistics.median(s), "p10": pick(0.10), "p90": pick(0.90), "min": s[0], "n": len(s)}

def pcc(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))

WT_BYTES = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}

def ckc(dev, fid, fp32):
    return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=FID[fid], math_approx_mode=False,
                                                  fp32_dest_acc_en=fp32, packer_l1_acc=True)

def time_trace(dev, op, reps, warm=5, samples=20):
    """Capture a trace of `reps` back-to-back op() calls (intermediate outputs freed inside the
    capture) and time blocking replays on the host. Returns (stats_us_per_replay, last_output)."""
    def free(o):
        for t in (o if isinstance(o, (list, tuple)) else [o]):
            ttnn.deallocate(t)
    w = op(); ttnn.synchronize_device(dev); free(w)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = None
    for r in range(reps):
        if out is not None:
            free(out)
        out = op()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    us = []
    for i in range(warm + samples):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        dt = (time.perf_counter() - t0) * 1e6
        if i >= warm:
            us.append(dt)
    return tid, stats(us), out

def per_op(dev, op, r_lo=1, r_hi=21, **kw):
    """Per-op device time as the slope between a 1-op and a 21-op trace replay (cancels the
    host round trip and launch), plus the 1-op replay latency and the 21-op average."""
    t1, s1, o1 = time_trace(dev, op, r_lo, **kw)
    ttnn.release_trace(dev, t1)
    for t in (o1 if isinstance(o1, (list, tuple)) else [o1]):
        ttnn.deallocate(t)
    th, sh, oh = time_trace(dev, op, r_hi, **kw)
    res = {
        "trace1_us": s1,
        f"trace{r_hi}_us": sh,
        "slope_us": (sh["median"] - s1["median"]) / (r_hi - r_lo),
        "avg_in_long_trace_us": sh["median"] / r_hi,
    }
    return th, res, oh

class Out:
    def __init__(self, path):
        self.f = open(path, "a")
    def emit(self, rec):
        line = json.dumps(rec)
        print("RESULT " + line, flush=True)
        self.f.write(line + "\n"); self.f.flush()
