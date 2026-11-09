"""pytest plugin (scratch, measurement only): wraps text_demo._run_traced_generation to add
per-step decode statistics (median/min/p90 over more steps) and a device-only trace replay time.
Loaded with `-p qwen_timing_plugin`; no tt-metal file is modified."""

import statistics
import time

import torch
from loguru import logger

EXTRA_STEPS = 150
RAW_REPLAYS = 100


def _make(mod):
    import ttnn
    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
    from models.tt_transformers.tt.generator import Generator

    def _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks):
        T = token_ids.shape[1]
        kv_cache_shape = [num_blocks, model.args.n_kv_heads, mod.BLOCK_SIZE, model.args.head_dim]
        model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
        page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        chunk_size = 2048
        model.capture_prefill_trace_chunked(
            device, page_table, chunk_size=chunk_size, warmup_masked_buckets=(T < chunk_size)
        )
        assert T < chunk_size
        t0 = time.time()
        logits = model.prefill_masked_bucket(token_ids, page_table, actual_len=T)
        logits_torch = ttnn.to_torch(logits).squeeze()
        next_token = logits_torch.argmax().item()
        ttft = time.time() - t0

        gen = Generator([model], [model.args], device)
        prime_decode_trace(gen, model, torch.tensor([[next_token]], dtype=torch.long), torch.tensor([T]), page_table)

        generated = [next_token]
        times = []
        current_pos = T
        total = max_generated_tokens - 1 + EXTRA_STEPS
        for i in range(total):
            t_step = time.time()
            out = gen.decode_forward(
                torch.tensor([[next_token]], dtype=torch.long),
                torch.tensor([current_pos]),
                page_table=page_table,
                kv_cache=None,
                enable_trace=True,
                read_from_device=True,
            )
            dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
            next_token = int(dl.argmax())
            times.append(time.time() - t_step)
            generated.append(next_token)
            current_pos += 1

        demo_steps = times[: max_generated_tokens - 1]
        ms = [t * 1000 for t in times]
        dms = [t * 1000 for t in demo_steps]
        logger.info(f"TIMING demo-window (49 steps): mean {statistics.mean(dms):.2f} ms, median {statistics.median(dms):.2f} ms")
        q = statistics.quantiles(ms, n=10)
        logger.info(
            f"TIMING all {len(ms)} host-timed steps: mean {statistics.mean(ms):.2f} median {statistics.median(ms):.2f} "
            f"min {min(ms):.2f} p10 {q[0]:.2f} p90 {q[-1]:.2f} ms"
        )
        logger.info(f"TIMING last 100 steps median {statistics.median(ms[-100:]):.2f} ms (positions {current_pos-100}..{current_pos})")

        # Device-only: back-to-back replays of the captured decode trace (no input copy, no readback).
        tids = gen.trace_ids_decode[False]
        mesh = model.mesh_device
        for _ in range(5):
            for _, tid in tids.items():
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(RAW_REPLAYS):
            for _, tid in tids.items():
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        raw = (time.time() - t0) / RAW_REPLAYS * 1000
        logger.info(f"TIMING device trace replay, {RAW_REPLAYS} back-to-back, one sync: {raw:.2f} ms/replay")
        per = []
        for _ in range(30):
            t0 = time.time()
            for _, tid in tids.items():
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            per.append((time.time() - t0) * 1000)
        logger.info(f"TIMING device trace replay with sync each: median {statistics.median(per):.2f} ms")

        text = tokenizer.decode(generated, skip_special_tokens=True)
        logger.info(f"TIMING full text ({len(generated)} tokens): {text!r}")
        avg = sum(demo_steps) / len(demo_steps)
        return generated[:max_generated_tokens], {"ttft": ttft, "avg_decode_s": avg, "decode_steps": len(demo_steps)}

    return _run_traced_generation


def pytest_collection_modifyitems(session, config, items):
    done = set()
    for item in items:
        mod = getattr(item, "module", None)
        if mod is not None and hasattr(mod, "_run_traced_generation") and id(mod) not in done:
            mod._run_traced_generation = _make(mod)
            done.add(id(mod))
            logger.info(f"qwen_timing_plugin: patched {mod.__name__}._run_traced_generation")
