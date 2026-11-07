"""Task 1: LoRA fine-tuning (custom loop, fp16 autocast, loss ONLY on the answer tokens).
Run on Colab:  python -m src.train_lora --out adapters/qwen15b_lora
Model selection uses VALIDATION loss only; the test split is never read here for training."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .common import BASE_MODEL, EOS_TEXT, fit_prompt, load_jsonl, parse_note


def build_example(tok, row, max_prompt_tokens):
    chat, dropped = fit_prompt(tok, parse_note(row["note"]), row["criterion_type"],
                               row["criterion_text"], max_prompt_tokens)
    gold = list(row["evidence_ids"])
    lost = [i for i in gold if i in dropped]          # gold evidence removed by truncation
    gold = [i for i in gold if i not in dropped]      # keep the target answerable
    target = json.dumps({"label": row["expert_eligibility"], "evidence_sentences": gold}) + EOS_TEXT
    p = tok(chat, add_special_tokens=False).input_ids
    t = tok(target, add_special_tokens=False).input_ids
    return {"input_ids": p + t, "labels": [-100] * len(p) + t, "n_prompt": len(p),
            "truncated": bool(dropped), "lost_gold": bool(lost)}


def length_stats(examples):
    L = np.array([len(e["input_ids"]) for e in examples])
    return {"n": len(L), "max": int(L.max()), "mean": float(L.mean()), "p95": float(np.percentile(L, 95)),
            "n_truncated": int(sum(e["truncated"] for e in examples)),
            "n_lost_gold_evidence": int(sum(e["lost_gold"] for e in examples))}


@torch.no_grad()
def val_loss(model, examples, dev):
    model.eval()
    tot = 0.0
    for e in examples:
        ids = torch.tensor([e["input_ids"]], device=dev)
        lab = torch.tensor([e["labels"]], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            tot += model(input_ids=ids, labels=lab).loss.item()
    model.train()
    return tot / len(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", default="adapters/qwen15b_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push_to_hub", default=None, help="HF repo id, e.g. username/trial-lora (needs HF_TOKEN)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base)

    train_rows, val_rows = load_jsonl("data/processed/train.jsonl"), load_jsonl("data/processed/val.jsonl")
    train = [build_example(tok, r, args.max_prompt_tokens) for r in train_rows]
    val = [build_example(tok, r, args.max_prompt_tokens) for r in val_rows]
    stats = {"train": length_stats(train), "val": length_stats(val)}
    print("Token-length / truncation stats:", json.dumps(stats, indent=1))
    Path("results/truncation_stats.json").write_text(json.dumps(stats, indent=2))

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=args.targets.split(","), task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)       # adapters are kept in fp32 by peft
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.accum)
    total = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total)), total)
    scaler = torch.amp.GradScaler("cuda")

    log = {"config": vars(args), "train": [], "val": [], "stats": stats}
    best, step, t0 = float("inf"), 0, time.time()
    model.train()
    for ep in range(args.epochs):
        order = list(range(len(train))); random.shuffle(order)
        run = []
        for i, idx in enumerate(order):
            e = train[idx]
            ids = torch.tensor([e["input_ids"]], device=dev)
            lab = torch.tensor([e["labels"]], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, labels=lab).loss
            run.append(loss.item())
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or i == len(order) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if step % args.log_every == 0:
                    m = float(np.mean(run)); run = []
                    log["train"].append({"step": step, "epoch": ep + (i + 1) / len(order), "loss": m})
                    print(f"ep {ep+1} step {step}/{total} train_loss {m:.4f} ({time.time()-t0:.0f}s)")
        vl = val_loss(model, val, dev)
        log["val"].append({"step": step, "epoch": ep + 1, "loss": vl})
        print(f"== epoch {ep+1}: val_loss {vl:.4f}")
        if vl < best:
            best = vl
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
            print("   saved best adapter ->", out / "best")
        Path("results/train_log.json").write_text(json.dumps(log, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot([x["epoch"] for x in log["train"]], [x["loss"] for x in log["train"]], label="train")
    plt.plot([x["epoch"] for x in log["val"]], [x["loss"] for x in log["val"]], "o-", label="validation")
    plt.xlabel("epoch"); plt.ylabel("loss (answer tokens)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("results/training_curve.png", dpi=200)
    print("best val loss:", best)

    if args.push_to_hub:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM as M
        base = M.from_pretrained(args.base, torch_dtype=torch.float16)
        PeftModel.from_pretrained(base, out / "best").push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)
        print("pushed adapter to", args.push_to_hub)


if __name__ == "__main__":
    main()
