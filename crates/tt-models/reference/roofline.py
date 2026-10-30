"""Rooflines on one Blackhole card for the model ladder's architectures past Llama: what a decode
step, a prefill and an image cannot beat, from the models' shapes and the card's ceilings.

Dev-only, standard library only: `python3 crates/tt-models/reference/roofline.py`. The tables in
`docs/plans/new-architectures.md` ("Roofline") are its output. SmolLM2-1.7B is computed first as
the calibration: the scoreboard's roofline (`docs/learnings/performance-scoreboard.md`) puts its
F32 decode step at 13.8 ms (6.85 GB of weights and 0.2 GB of cache at 512 GB/s).

The card (`docs/learnings/dataflow-compiler.md` T1, T2, T4; the scoreboard's ceilings):
- GDDR6: 512 GB/s peak; 488 measured, the read ceiling.
- The matrix unit: 4,096 FLOP a cycle a tile a fidelity phase, 120 tiles at 1.35 GHz: 664 TFLOPS
  at one phase (LoFi, lossy for these operands), 332 at two, 166 at four (HiFi4, what an exact
  product of F32 (TF32) or BF16 operands takes). Derived, not measured; the best measured kernel
  is 35.5 TFLOPS with an FP32 `B` and 49.5 with BF16 (F4b.2, per product).
- 32 GiB of GDDR6 a card.

What a step must move: every weight it multiplies by, once (a decode step reads each matrix
once; the embedding tables give one row a token), the cache it attends to, and the recurrent
state it reads and writes. Activations are ignored for the language models (kilobytes at batch 1)
and counted for the image model. FLOPs are the products' `2 m k n` and attention's `4 n_q n_kv d`
a head; element-wise work is listed but not charged, because it runs on the SFPU, whose ceiling is
not a matrix-unit ceiling (see the notes the script prints).
"""

PEAK_BW = 512e9
READ_BW = 488e9
TFLOPS = {"HiFi4": 166e12, "HiFi2": 332e12, "LoFi": 664e12}
MEASURED = {"F32": 35.5e12, "BF16": 49.5e12}
GDDR_BYTES = 32 * 2**30

# Bytes an element: BFP formats hold a shared exponent byte for each 16 values.
BYTES = {"F32": 4.0, "BF16": 2.0, "BFP8": 1 + 1 / 16, "BFP4": 0.5 + 1 / 16}


def ms(seconds):
    return seconds * 1e3


# ---------------------------------------------------------------------------------------------
# Shapes: (matrix parameters a step multiplies by, embedding-table parameters, per-token cache
# bytes a position, fixed state bytes, FLOPs a token besides the products, attention FLOPs).


def llama(hidden, heads, kv_heads, head_dim, ff, vocab, layers):
    per_layer = hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim \
        + heads * head_dim * hidden + 3 * hidden * ff
    return dict(
        matrices=layers * per_layer + vocab * hidden,  # the tied head is a product
        table_row=hidden,
        kv_per_position=layers * 2 * kv_heads * head_dim,  # elements
        windowed_kv_per_position=0,
        window=0,
        state=0,
        attention_flops_per_position=layers * 4 * heads * head_dim,
        prefill_matrices=layers * per_layer,
        head=vocab * hidden,
    )


def qwen35(hidden, ff, layers=32, vocab=248_320, heads=16, kv_heads=4, head_dim=256,
           k_heads=16, k_dim=128, v_heads=32, v_dim=128, conv=4, every=4):
    full = sum(1 for i in range(layers) if (i + 1) % every == 0)
    linear = layers - full
    key_dim, value_dim = k_heads * k_dim, v_heads * v_dim
    mlp = 3 * hidden * ff
    delta = hidden * (2 * key_dim + value_dim) + hidden * value_dim + 2 * hidden * v_heads \
        + value_dim * hidden
    attn = hidden * 2 * heads * head_dim + 2 * hidden * kv_heads * head_dim \
        + heads * head_dim * hidden
    per = linear * (delta + mlp) + full * (attn + mlp)
    return dict(
        matrices=per + vocab * hidden,
        table_row=hidden,
        kv_per_position=full * 2 * kv_heads * head_dim,
        windowed_kv_per_position=0,
        window=0,
        # The recurrent state, F32, read and written every step, and the convolution's inputs.
        state=linear * (2 * v_heads * k_dim * v_dim * 4 + 2 * (conv - 1) * (2 * key_dim + value_dim) * 4),
        state_elementwise=linear * v_heads * k_dim * v_dim * 4,  # decay, update: SFPU passes
        attention_flops_per_position=full * 4 * heads * head_dim,
        recurrence_flops=linear * v_heads * 6 * k_dim * v_dim,
        prefill_matrices=per,
        head=vocab * hidden,
        linear_layers=linear,
    )


def gemma4(hidden=2560, heads=8, layers=42, every=6, shared=18, ff=10240, vocab=262_144,
           per_layer=256, sliding=(256, 2), full=(512, 2), window=512):
    first_shared = layers - shared
    mats, kv_full, kv_slide, attn_full, attn_slide = 0, 0, 0, 0, 0
    for i in range(layers):
        is_full = (i + 1) % every == 0
        hd, kvh = full if is_full else sliding
        mats += hidden * heads * hd + heads * hd * hidden + 3 * hidden * ff \
            + 2 * hidden * per_layer
        if i < first_shared:
            mats += 2 * hidden * kvh * hd
            if is_full:
                kv_full += 2 * kvh * hd
            else:
                kv_slide += 2 * kvh * hd
        if is_full:
            attn_full += 4 * heads * hd
        else:
            attn_slide += 4 * heads * hd
    mats += hidden * layers * per_layer  # the per-layer model projection
    return dict(
        matrices=mats + vocab * hidden,
        table_row=hidden + layers * per_layer,  # a token's row of each table
        kv_per_position=kv_full,
        windowed_kv_per_position=kv_slide,
        window=window,
        state=0,
        attention_flops_per_position=attn_full,
        windowed_attention_flops_per_position=attn_slide,
        prefill_matrices=mats,
        head=vocab * hidden,
        per_layer_table=vocab * layers * per_layer,
    )


def decode(name, m, positions, weights, cache="F32", bw=READ_BW):
    w = BYTES[weights]
    weight_bytes = m["matrices"] * w + m["table_row"] * 2  # tables in BF16
    kv = m["kv_per_position"] * positions
    if m["windowed_kv_per_position"]:
        kv += m["windowed_kv_per_position"] * min(positions, m["window"])
    cache_bytes = kv * BYTES[cache] + m["state"]
    total = weight_bytes + cache_bytes
    flops = 2 * m["matrices"] + m["attention_flops_per_position"] * positions \
        + m.get("windowed_attention_flops_per_position", 0) * min(positions, m.get("window", 0) or positions) \
        + m.get("recurrence_flops", 0)
    t_mem = total / bw
    t_hifi = flops / TFLOPS["HiFi4"]
    return dict(name=name, weights=weights, positions=positions, gb=total / 1e9,
                weight_gb=weight_bytes / 1e9, cache_gb=cache_bytes / 1e9, gflop=flops / 1e9,
                ms_peak=ms(total / PEAK_BW), ms_read=ms(t_mem), ms_compute=ms(t_hifi),
                intensity=flops / total)


def prefill(name, m, tokens, weights):
    w = BYTES[weights]
    flops = 2 * m["prefill_matrices"] * tokens + 2 * m["head"]  # the head for the last token only
    # Causal attention over the prompt: each query attends to its prefix (sliding: its window).
    att = m["attention_flops_per_position"] * tokens * (tokens + 1) / 2
    if m.get("windowed_attention_flops_per_position"):
        wnd = m["window"]
        pairs = sum(min(i + 1, wnd) for i in range(tokens))
        att += m["windowed_attention_flops_per_position"] * pairs
    flops += att + m.get("recurrence_flops", 0) * tokens
    bytes_ = m["matrices"] * w
    return dict(name=name, tokens=tokens, tflop=flops / 1e12,
                ms_compute=ms(flops / TFLOPS["HiFi4"]),
                ms_measured_kernel=ms(flops / MEASURED["F32" if weights == "F32" else "BF16"]),
                ms_weights=ms(bytes_ / READ_BW))


# ---------------------------------------------------------------------------------------------
# MOSS-Image.

def dit_params(dim=3072, ff=8192, layers=42, text_dim=4096, latent=128, freq=256):
    block = 4 * dim * dim + dim * 2 * ff + ff * dim
    text_block = 3 * dim * dim + dim * dim + 3 * dim * ff
    text = text_dim * dim + 2 * text_block
    small = latent * dim + freq * dim + dim * dim + dim * 6 * dim + dim * latent + dim * 2 * dim
    return dict(block=block, layers=layers, text=text, text_block=text_block, small=small,
                total=layers * block + text + small, dim=dim, ff=ff, text_dim=text_dim)


def dit_forward(p, image_tokens, text_tokens):
    n = image_tokens + text_tokens
    dim = p["dim"]
    mm = 2 * p["layers"] * p["block"] * n
    attn = p["layers"] * 4 * n * n * dim
    text = 2 * p["text_dim"] * dim * text_tokens + 2 * (2 * p["text_block"] * text_tokens
                                                         + 4 * text_tokens ** 2 * dim)
    return dict(flops=mm + attn + text, matmul=mm, attention=attn)


def vae_decode(size, latent=32, widths=(384, 384, 192, 96), layers_per_block=2):
    """FLOPs and the activations' minimal traffic (each op reads its input and writes its output
    once, F32) of the FLUX.2 decoder producing a `size` x `size` image."""
    h = size // 8
    flops, traffic = 0, 0

    def conv(cin, cout, k, side):
        nonlocal flops, traffic
        flops += 2 * k * k * cin * cout * side * side
        traffic += 4 * side * side * (cin + cout)

    def elementwise(c, side, passes):
        nonlocal traffic
        traffic += 4 * side * side * c * 2 * passes

    def resnet(cin, cout, side):
        elementwise(cin, side, 2)  # norm1, silu
        conv(cin, cout, 3, side)
        elementwise(cout, side, 2)  # norm2, silu
        conv(cout, cout, 3, side)
        if cin != cout:
            conv(cin, cout, 1, side)
        elementwise(cout, side, 1)  # the residual sum

    conv(latent, latent, 1, h)  # post_quant_conv
    conv(latent, widths[0], 3, h)
    resnet(widths[0], widths[0], h)
    pixels = h * h
    flops += 4 * pixels * pixels * widths[0] + 3 * 2 * pixels * widths[0] ** 2 \
        + 2 * pixels * widths[0] ** 2
    traffic += 4 * pixels * widths[0] * 6 + 4 * pixels * pixels  # q, k, v, out; the scores once
    resnet(widths[0], widths[0], h)
    side, prev = h, widths[0]
    for i, out in enumerate(widths):
        for j in range(layers_per_block + 1):
            resnet(prev if j == 0 else out, out, side)
        prev = out
        if i + 1 < len(widths):
            side *= 2
            elementwise(out, side, 1)  # the upsampling's moves
            conv(out, out, 3, side)
    elementwise(widths[-1], side, 2)
    conv(widths[-1], 3, 3, side)
    return dict(flops=flops, traffic=traffic)


def table(rows, cols):
    head = "| " + " | ".join(c for c, _ in cols) + " |"
    rule = "|" + "|".join("---" if i == 0 else "--:" for i in range(len(cols))) + "|"
    print(head)
    print(rule)
    for r in rows:
        print("| " + " | ".join(f(r) for _, f in cols) + " |")
    print()


def main():
    models = {
        "SmolLM2-1.7B (calibration)": llama(2048, 32, 32, 64, 8192, 49_152, 24),
        "Qwen3.5-4B": qwen35(2560, 9216),
        "Nemotron-3.5-Bolt-E4B": gemma4(),
        "MOSS text encoder": qwen35(4096, 12288),
    }
    print("## Parameters a step multiplies by\n")
    for name, m in models.items():
        print(f"- {name}: {m['matrices'] / 1e9:.3f} B in products (the head {m['head'] / 1e9:.3f} B)"
              + (f"; per-layer table {m['per_layer_table'] / 1e9:.2f} B, one row a token"
                 if "per_layer_table" in m else "")
              + (f"; recurrent state {m['state'] / 1e6:.1f} MB read and written a step"
                 if m["state"] else ""))
    print()

    print("## Decode, batch 1 (one token a step)\n")
    cols = [
        ("Model", lambda r: r["name"]),
        ("Weights", lambda r: r["weights"]),
        ("Context", lambda r: f"{r['positions']:,}"),
        ("GB a step", lambda r: f"{r['gb']:.2f}"),
        ("of it cache/state", lambda r: f"{r['cache_gb']:.3f}"),
        ("ms at 512 GB/s", lambda r: f"{r['ms_peak']:.2f}"),
        ("ms at 488 (read ceiling)", lambda r: f"{r['ms_read']:.2f}"),
        ("tokens/s at 488", lambda r: f"{1e3 / r['ms_read']:.0f}"),
        ("ms of HiFi4 compute", lambda r: f"{r['ms_compute']:.3f}"),
        ("FLOP/byte", lambda r: f"{r['intensity']:.2f}"),
    ]
    rows = [decode("SmolLM2-1.7B (calibration)", models["SmolLM2-1.7B (calibration)"], 512, "F32")]
    for name in ["Qwen3.5-4B", "Nemotron-3.5-Bolt-E4B"]:
        for weights in ["F32", "BF16", "BFP8", "BFP4"]:
            for positions in [512, 8192]:
                rows.append(decode(name, models[name], positions, weights))
    table(rows, cols)

    print("## Prefill (compute-bound): a prompt of N tokens\n")
    cols = [
        ("Model", lambda r: r["name"]),
        ("Tokens", lambda r: f"{r['tokens']:,}"),
        ("TFLOP", lambda r: f"{r['tflop']:.2f}"),
        ("ms at 166 TFLOPS (HiFi4)", lambda r: f"{r['ms_compute']:.1f}"),
        ("ms at the measured kernel", lambda r: f"{r['ms_measured_kernel']:.1f}"),
        ("ms to read the weights once", lambda r: f"{r['ms_weights']:.1f}"),
    ]
    rows = []
    for name in ["Qwen3.5-4B", "Nemotron-3.5-Bolt-E4B"]:
        for n in [512, 2048, 8192]:
            rows.append(prefill(name, models[name], n, "BF16"))
    table(rows, cols)

    print("## MOSS-Image, one image\n")
    p = dit_params()
    print(f"- DiT: {p['total'] / 1e9:.3f} B parameters ({p['layers']} blocks of "
          f"{p['block'] / 1e6:.1f} M, text embedder {p['text'] / 1e6:.1f} M)")
    enc = models["MOSS text encoder"]
    enc_tokens = 17 + 10  # the prompt and the empty negative prompt, chat-templated
    enc_bytes = enc["prefill_matrices"] * 2
    enc_flops = 2 * enc["prefill_matrices"] * enc_tokens
    print(f"- Text encoder: {enc['prefill_matrices'] / 1e9:.2f} B in products; encoding a prompt and "
          f"the negative one ({enc_tokens} tokens) reads {enc_bytes / 1e9:.1f} GB of BF16 weights at "
          f"least once: {ms(enc_bytes / READ_BW):.0f} ms at 488 GB/s ({enc_flops / 1e12:.2f} TFLOP, "
          f"{ms(enc_flops / TFLOPS['HiFi4']):.1f} ms of HiFi4) - once a prompt, not a step")
    print()
    cols = [
        ("Resolution", lambda r: r["res"]),
        ("Image tokens", lambda r: f"{r['tokens']:,}"),
        ("TFLOP a forward", lambda r: f"{r['tflop']:.1f}"),
        ("of it attention", lambda r: f"{r['attn']:.0%}"),
        ("ms a forward at 166 TFLOPS", lambda r: f"{r['ms']:.0f}"),
        ("ms to read the BF16 weights", lambda r: f"{r['ms_w']:.0f}"),
        ("s an image, 50 steps x 2 (CFG), HiFi4", lambda r: f"{r['image_s']:.1f}"),
        ("at the measured BF16 kernel", lambda r: f"{r['image_measured']:.1f}"),
        ("at LoFi (lossy)", lambda r: f"{r['image_lofi']:.1f}"),
        ("VAE decode TFLOP", lambda r: f"{r['vae_tflop']:.2f}"),
        ("VAE ms, compute / traffic", lambda r: f"{r['vae_ms']:.0f} / {r['vae_bw_ms']:.0f}"),
    ]
    rows = []
    for res in [256, 512, 1024]:
        tokens = (res // 16) ** 2
        f = dit_forward(p, tokens, 17)
        v = vae_decode(res)
        forward = f["flops"] / TFLOPS["HiFi4"]
        rows.append(dict(
            res=f"{res} x {res}", tokens=tokens, tflop=f["flops"] / 1e12,
            attn=f["attention"] / f["flops"], ms=ms(forward),
            ms_w=ms(p["total"] * 2 / READ_BW),
            image_s=100 * max(forward, p["total"] * 2 / READ_BW),
            image_measured=100 * f["flops"] / MEASURED["BF16"],
            image_lofi=100 * max(f["flops"] / TFLOPS["LoFi"], p["total"] * 2 / READ_BW),
            vae_tflop=v["flops"] / 1e12, vae_ms=ms(v["flops"] / TFLOPS["HiFi4"]),
            vae_bw_ms=ms(v["traffic"] / READ_BW)))
    table(rows, cols)

    print("## Memory on one card (32 GiB)\n")
    q = models["Qwen3.5-4B"]
    g = models["Nemotron-3.5-Bolt-E4B"]
    for name, n, extra in [
        ("Qwen3.5-4B", q["matrices"] - q["head"] + q["head"], 0),
        ("Nemotron-3.5-Bolt-E4B (tables BF16)", g["matrices"] - g["head"], (g["head"] + g["per_layer_table"]) * 2),
        ("MOSS DiT", p["total"], 0),
        ("MOSS text encoder (table BF16)", enc["matrices"] - enc["head"], enc["head"] * 2),
    ]:
        for wname in ["F32", "BF16"]:
            gb = (n * BYTES[wname] + extra) / 2**30
            print(f"- {name}, {wname} weights: {gb:.1f} GiB"
                  + (" (does not fit)" if gb > 32 else ""))
    print()


if __name__ == "__main__":
    main()
