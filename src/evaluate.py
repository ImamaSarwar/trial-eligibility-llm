"""Tasks 1/2: run base or fine-tuned model on val/test with greedy decoding, save predictions + metrics.
  python -m src.evaluate --split val  --tag base
  python -m src.evaluate --split val  --tag ft --adapter adapters/qwen15b_lora/best
  python -m src.evaluate --split test --tag ft --adapter ... --confirm_test      (do this ONCE at the end)"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import BASE_MODEL, LABELS, fit_prompt, load_jsonl, parse_note, parse_output
from .metrics import compute_all, print_summary


def load_model(base, adapter, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--tag", required=True, help="base | ft | any name")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_prompt_tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N rows")
    ap.add_argument("--confirm_test", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("The test split must be evaluated once, at the very end. Add --confirm_test.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    model = load_model(args.base, args.adapter, dtype, device)

    rows = load_jsonl(f"data/processed/{args.split}.jsonl")[: args.limit]
    items = []
    for r in rows:
        chat, dropped = fit_prompt(tok, parse_note(r["note"]), r["criterion_type"], r["criterion_text"],
                                   args.max_prompt_tokens)
        items.append((r, chat, dropped))
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))      # similar lengths per batch
    raw = {}
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        enc = tok([items[i][1] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            raw[i] = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"{min(b + args.batch_size, len(order))}/{len(order)}")

    records = []
    for i, (r, chat, dropped) in enumerate(items):
        p = parse_output(raw[i], r["criterion_type"])
        records.append({
            "annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
            "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
            "gold_label": r["expert_eligibility"], "gold_ids": r["evidence_ids"],
            "pred_label": p["label"] if p["label"] in LABELS[r["criterion_type"]] else "invalid",
            "pred_ids": p["evidence"], "parse_ok": p["ok"], "raw_output": raw[i],
            "truncated": bool(dropped), "lost_gold": any(g in dropped for g in r["evidence_ids"]),
        })
    Path("results").mkdir(exist_ok=True)
    stem = f"{args.tag}_{args.split}"
    with open(f"results/preds_{stem}.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    m = compute_all(records)
    m["settings"] = {"base": args.base, "adapter": args.adapter, "decoding": "greedy (do_sample=False)",
                     "max_new_tokens": args.max_new_tokens, "dtype": str(dtype), "batch_size": args.batch_size,
                     "max_prompt_tokens": args.max_prompt_tokens, "n_truncated": sum(x["truncated"] for x in records)}
    Path(f"results/metrics_{stem}.json").write_text(json.dumps(m, indent=2))
    print_summary(m, stem)


if __name__ == "__main__":
    main()
