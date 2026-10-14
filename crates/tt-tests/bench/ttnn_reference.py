#!/usr/bin/env python3
"""tt-metal (ttnn) reference timings on one card, for the firmware-architecture review.

Development only: it is a measuring stick for `cargo xtask bench`, not part of any
shipped build. Run it with the installed ttnn and nothing else on the card:

    ~/.tenstorrent-venv/bin/python -I crates/tt-tests/bench/ttnn_reference.py --device 0

Same shapes as the firmware benches (`silicon_bench_firmware`, `silicon_bench_path`):
element-wise add, matmul and a one-token embedding lookup. Each is timed three ways, on
the host with `time.perf_counter` after `synchronize_device`:

  eager_sync   one op, then synchronize: the host round trip of a single launch
  eager_batch  16 ops queued, then one synchronize, per op: the enqueue-bound rate
  trace        `execute_trace(blocking=True)` of a captured single op: replay

3 warmups, 15 samples, median / p10 / p90. Every output is checked against torch on
the host; the check is recorded (exactness or the measured error), never used to pick a
tolerance. Results go to stdout as `BENCH {json}` lines in the shape `tt_tests::bench`
emits, and to `target/silicon/bench/ttnn-<stamp>.jsonl`.
"""
import argparse
import json
import os
import statistics
import subprocess
import time

import torch
import ttnn

WARMUP = 3
SAMPLES = 15
BATCH = 16


def stats(us):
    s = sorted(us)
    pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]
    return {"median": statistics.median(s), "p10": pick(0.10), "p90": pick(0.90), "n": len(s)}


class Out:
    def __init__(self, path):
        self.f = open(path, "w")

    def emit(self, key, unit, timed, st, **extra):
        rec = {"kind": "result", "key": key, "unit": unit, "timed": timed, **st, **extra}
        line = json.dumps(rec)
        print("BENCH " + line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def note(self, key, **fields):
        rec = {"kind": "note", "key": key, **fields}
        print("NOTE " + json.dumps(rec), flush=True)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


def time_modes(dev, out, name, op, check):
    """Time `op()` (which returns a device tensor) three ways; check one output."""
    first = op()
    ttnn.synchronize_device(dev)
    out.note(name + " check", **check(ttnn.to_torch(first)))
    ttnn.deallocate(first)

    # eager, synchronized each time
    us = []
    for i in range(WARMUP + SAMPLES):
        t0 = time.perf_counter()
        r = op()
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e6
        ttnn.deallocate(r)
        if i >= WARMUP:
            us.append(dt)
    out.emit(name + " eager_sync", "us", "host", stats(us))

    # eager, batched
    us = []
    for i in range(WARMUP + SAMPLES):
        t0 = time.perf_counter()
        rs = [op() for _ in range(BATCH)]
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e6 / BATCH
        for r in rs:
            ttnn.deallocate(r)
        if i >= WARMUP:
            us.append(dt)
    out.emit(name + " eager_batch", "us", "host", stats(us))

    # trace replay
    try:
        warm = op()  # compile outside the capture
        ttnn.synchronize_device(dev)
        ttnn.deallocate(warm)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        traced = op()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        us = []
        for i in range(WARMUP + SAMPLES):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            dt = (time.perf_counter() - t0) * 1e6
            if i >= WARMUP:
                us.append(dt)
        # The replayed output must still be the op's.
        out.note(name + " trace check", **check(ttnn.to_torch(traced)))
        ttnn.release_trace(dev, tid)
        out.emit(name + " trace", "us", "host", stats(us))
    except Exception as e:  # a trace that cannot be captured is a result too
        out.note(name + " trace", skipped=str(e).splitlines()[0][:200])


def exact_or_error(ref):
    def check(got):
        got = got.to(torch.float64)
        refd = ref.to(torch.float64)
        diff = (got - refd).abs()
        return {
            "bit_exact": bool(torch.equal(got.to(ref.dtype), ref)),
            "max_abs_err": float(diff.max()),
            "max_rel_err_of_max": float(diff.max() / refd.abs().max().clamp_min(1e-300)),
        }

    return check


def to_dev(dev, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--only", default="", help="substring filter on case names")
    a = ap.parse_args()

    stamp = int(time.time())
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    os.makedirs(os.path.join(root, "target", "silicon", "bench"), exist_ok=True)
    out = Out(os.path.join(root, "target", "silicon", "bench", f"ttnn-{stamp}.jsonl"))
    sha = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    out.note("conditions", card=a.device, git=sha, warmup=WARMUP, samples=SAMPLES, batch=BATCH,
             torch=torch.__version__)

    dev = ttnn.open_device(device_id=a.device, trace_region_size=64 << 20)
    try:
        g = torch.Generator().manual_seed(7)
        want = lambda n: a.only in n

        # element-wise add, F32 (what the firmware benches add)
        for side in (64, 2048):
            name = f"ttnn add f32 {side}x{side}"
            if not want(name):
                continue
            x = torch.randn(side, side, generator=g)
            y = torch.randn(side, side, generator=g)
            tx, ty = to_dev(dev, x, ttnn.float32), to_dev(dev, y, ttnn.float32)
            time_modes(dev, out, name, lambda: ttnn.add(tx, ty), exact_or_error(x + y))
            ttnn.deallocate(tx)
            ttnn.deallocate(ty)

        # matmul, F32 and BF16
        for n in (256, 1024):
            for label, tdt in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
                name = f"ttnn matmul {label} {n}x{n}x{n}"
                if not want(name):
                    continue
                x = torch.randn(n, n, generator=g)
                y = torch.randn(n, n, generator=g)
                if label == "bf16":
                    x, y = x.bfloat16().float(), y.bfloat16().float()
                tx, ty = to_dev(dev, x, tdt), to_dev(dev, y, tdt)
                ref = x.double() @ y.double()
                time_modes(dev, out, name, lambda: ttnn.matmul(tx, ty), exact_or_error(ref))
                ttnn.deallocate(tx)
                ttnn.deallocate(ty)

        # one-token embedding lookup from SmolLM2's table shape
        name = "ttnn embedding 49152x576 1 token"
        if want(name):
            for label, tdt, ttorch in (("f32", ttnn.float32, torch.float32),
                                       ("bf16", ttnn.bfloat16, torch.bfloat16)):
                try:
                    table = torch.randn(49152, 576, generator=g).to(ttorch)
                    ids = torch.tensor([[1234]], dtype=torch.int32)
                    tt_table = to_dev(dev, table, tdt, ttnn.ROW_MAJOR_LAYOUT)
                    tt_ids = to_dev(dev, ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                    ref = table[1234].reshape(1, 1, 576)
                    time_modes(
                        dev, out, f"{name} {label}",
                        lambda: ttnn.embedding(tt_ids, tt_table),
                        lambda got: exact_or_error(ref)(got.reshape(1, 1, -1)[..., :576]),
                    )
                    ttnn.deallocate(tt_table)
                    ttnn.deallocate(tt_ids)
                except Exception as e:
                    out.note(f"{name} {label}", skipped=str(e).splitlines()[0][:200])
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
