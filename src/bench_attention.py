"""Task 4 benchmark: reference vs tiled (online softmax) vs PyTorch SDPA. Saves results/attention_bench.json
  python -m src.bench_attention"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .attention import reference_attention, tiled_attention


def timeit(fn, dev, reps=5):
    fn()                                           # warm-up
    ts = []
    for _ in range(reps):
        if dev == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter(); fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None
    return sum(ts) / len(ts) * 1000, mem


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for N in (128, 512, 1024, 2048):
        for causal in (False, True):
            torch.manual_seed(0)
            B, H, D = 1, 8, 64
            q, k, v = (torch.randn(B, H, N, D, device=dev) for _ in range(3))
            ref = reference_attention(q, k, v, causal)
            til = tiled_attention(q, k, v, causal, 128, 128)
            sd = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            r = {"N": N, "causal": causal, "device": dev,
                 "tiled_max_err": (ref - til).abs().max().item(), "tiled_mean_err": (ref - til).abs().mean().item(),
                 "sdpa_max_err": (ref - sd).abs().max().item()}
            for name, fn in [("reference", lambda: reference_attention(q, k, v, causal)),
                             ("tiled", lambda: tiled_attention(q, k, v, causal, 128, 128)),
                             ("sdpa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))]:
                ms, mem = timeit(fn, dev)
                r[f"{name}_ms"], r[f"{name}_peak_MB"] = ms, mem
            rows.append(r)
            print({k_: (round(v_, 6) if isinstance(v_, float) else v_) for k_, v_ in r.items()})
    Path("results").mkdir(exist_ok=True)
    Path("results/attention_bench.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
