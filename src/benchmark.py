"""Tasks 3 + 5 benchmark: no-cache vs KV-cache vs speculative (and HF generate / HF assisted as references).
  python -m src.benchmark --tag t4_fp16
Greedy decoding, eos disabled so every mode produces exactly n_new tokens. Warm-up + synchronized repeats."""
import argparse
import json
import platform
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import BASE_MODEL, DRAFT_MODEL, load_jsonl
from .kv_decode import generate_no_cache, generate_with_cache, kv_cache_bytes, sync
from .speculative import speculative_generate


def make_input(tok, n_tokens, device):
    """Real clinical text (validation notes) cut to an exact token length."""
    text = " ".join(r["note"] for r in load_jsonl("data/processed/val.jsonl")[:80])
    ids = tok(text, add_special_tokens=False).input_ids
    while len(ids) < n_tokens:
        ids = ids + ids
    return torch.tensor([ids[:n_tokens]], device=device)


def hf_generate(model, ids, n_new, assistant=None):
    sync(ids.device); t = time.perf_counter()
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new, min_new_tokens=n_new,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         assistant_model=assistant, pad_token_id=0)
    sync(ids.device)
    return {"tokens": out[0, ids.shape[1]:].tolist(), "prefill_s": None, "decode_s": time.perf_counter() - t}


def run_mode(fn, dev, reps):
    fn()                                                   # warm-up (not recorded)
    runs = []
    for _ in range(reps):
        if dev == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        r = fn()
        r["peak_MB"] = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None
        runs.append(r)
    return runs


def summarize(runs, n_new):
    def ms(key):
        v = [r[key] for r in runs if r.get(key) is not None]
        return (st.mean(v), st.pstdev(v)) if v else (None, None)
    total = [(r["prefill_s"] or 0) + r["decode_s"] for r in runs]
    return {"prefill_s": ms("prefill_s"), "decode_s": ms("decode_s"),
            "total_s": (st.mean(total), st.pstdev(total)),
            "tokens_per_s": n_new / st.mean(total),
            "decode_tokens_per_s": (n_new - 1) / st.mean([r["decode_s"] for r in runs]) if runs[0]["prefill_s"] else None,
            "peak_MB": ms("peak_MB")[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--draft", default=DRAFT_MODEL)
    ap.add_argument("--adapter", default=None, help="merge this LoRA into the TARGET before benchmarking")
    ap.add_argument("--configs", default="128:32,512:64,1024:128", help="prompt_len:new_tokens,...")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--skip_hf", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base)
    target = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        target = PeftModel.from_pretrained(target, args.adapter).merge_and_unload()
    target = target.to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype).to(dev).eval()
    if tok.get_vocab() != AutoTokenizer.from_pretrained(args.draft).get_vocab():
        raise SystemExit('Draft and target tokenizers differ - speculative decoding needs identical tokenization')
    if dev == "cuda":
        torch.cuda.synchronize(); baseline = torch.cuda.memory_allocated() / 2**20
    else:
        baseline = None

    env = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
           "dtype": str(dtype), "torch": torch.__version__, "k": args.k, "reps": args.reps,
           "target": args.base, "draft": args.draft, "adapter": args.adapter, "weights_MB": baseline}
    results = []
    for cfg in args.configs.split(","):
        L, n_new = map(int, cfg.split(":"))
        ids = make_input(tok, L, dev)
        modes = {
            "no_cache": lambda: generate_no_cache(target, ids, n_new),
            "kv_cache": lambda: generate_with_cache(target, ids, n_new),
            "speculative": lambda: speculative_generate(target, draft, ids, n_new, k=args.k),
        }
        if not args.skip_hf:
            modes["hf_generate"] = lambda: hf_generate(target, ids, n_new)
            modes["hf_assisted"] = lambda: hf_generate(target, ids, n_new, assistant=draft)
        ref_tokens = None
        for name, fn in modes.items():
            runs = run_mode(fn, dev, args.reps)
            toks = runs[0]["tokens"]
            if name == "no_cache":
                ref_tokens = toks
            first_diff = next((i for i, (a, b) in enumerate(zip(toks, ref_tokens)) if a != b), None)
            row = {"prompt_len": L, "new_tokens": n_new, "mode": name, **summarize(runs, n_new),
                   "identical_to_no_cache": toks == ref_tokens, "first_mismatch_index": first_diff}
            if name == "speculative":
                s = runs[0]["stats"]
                row.update(acceptance_rate=s["acceptance_rate"], tokens_per_target_pass=s["tokens_per_target_pass"],
                           target_forwards=s["target_forwards"], accepted_per_round=s["accepted_per_round"])
            if name == "kv_cache":
                row["kv_cache_MB_analytic"] = kv_cache_bytes(target.config, L + n_new, 2 if dev == "cuda" else 4) / 2**20
            results.append(row)
            print(f"{L:>5}/{n_new:<4} {name:12s} total {row['total_s'][0]:.3f}s  {row['tokens_per_s']:.1f} tok/s  "
                  f"equal={row['identical_to_no_cache']}  peakMB={row['peak_MB']}")
    Path("results").mkdir(exist_ok=True)
    out = Path(f"results/benchmark_{args.tag}.json")
    out.write_text(json.dumps({"env": env, "results": results}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
