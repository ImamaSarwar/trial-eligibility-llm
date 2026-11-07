"""Error analysis + confusion-matrix figure.
  python -m src.error_analysis --tag ft --split val
Writes results/error_analysis_<tag>_<split>.md and results/confusion_<tag>_<split>.png"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import load_jsonl, parse_note

POLARITY = {("included", "not included"), ("not included", "included"),
            ("excluded", "not excluded"), ("not excluded", "excluded")}


def category(g, p):
    if p == "invalid":
        return "unparsable / invalid output"
    if (g, p) in POLARITY:
        return "polarity flip (direct contradiction of expert)"
    if "not enough information" in (g, p):
        return "missing-information confusion (insufficient info vs decision)"
    if "not applicable" in (g, p):
        return "applicability confusion (not applicable vs decision)"
    return "other"


def ev_text(sents, ids):
    return " ".join(f"[{i}] {sents[i]}" for i in ids if i in sents) or "(none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_examples", type=int, default=12)
    a = ap.parse_args()
    stem = f"{a.tag}_{a.split}"
    preds = load_jsonl(f"results/preds_{stem}.jsonl")
    data = {r["annotation_id"]: r for r in load_jsonl(f"data/processed/{a.split}.jsonl")}
    m = json.load(open(f"results/metrics_{stem}.json"))

    # ---- per-class table (paste this into the report) ----
    for part in ("overall", "inclusion", "exclusion"):
        if part not in m:
            continue
        print(f"\n[{part}] per-class precision / recall / F1 / support")
        for lab, d in m[part]["classification"]["per_class"].items():
            print(f"  {lab:24s} P={d['precision']:.2f} R={d['recall']:.2f} F1={d['f1-score']:.2f} n={int(d['support'])}")

    # ---- confusion matrices ----
    parts = [p for p in ("overall", "inclusion", "exclusion") if p in m]
    fig, axes = plt.subplots(1, len(parts), figsize=(5.2 * len(parts), 4.6))
    for ax, part in zip(axes if len(parts) > 1 else [axes], parts):
        c = m[part]["classification"]
        cm, labs = c["confusion_matrix"], c["labels"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(labs, fontsize=7)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
        ax.set_title(f"{part} (rows=expert, cols=model)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"results/confusion_{stem}.png", dpi=200)

    # ---- errors ----
    errors = [p for p in preds if p["pred_label"] != p["gold_label"]]
    cat_counts = Counter(category(e["gold_label"], e["pred_label"]) for e in errors)
    pair_counts = Counter((e["gold_label"], e["pred_label"]) for e in errors)
    wrong_ev = [p for p in preds if p["pred_label"] == p["gold_label"] and set(p["pred_ids"]) != set(p["gold_ids"])]
    print(f"\n{len(errors)} label errors out of {len(preds)}")
    for k, v in cat_counts.most_common():
        print(f"  {v:3d}  {k}")
    print("most common (expert -> model):", pair_counts.most_common(6))
    print(f"correct label but different evidence: {len(wrong_ev)}")

    # pick diverse examples: round-robin over categories, distinct (gold,pred) pairs first
    by_cat = defaultdict(list)
    seen = set()
    for e in sorted(errors, key=lambda e: e["annotation_id"]):
        key = (e["gold_label"], e["pred_label"])
        by_cat[category(*key)].append((key in seen, e)); seen.add(key)
    picked, cats = [], list(by_cat)
    for c in cats:
        by_cat[c].sort(key=lambda t: t[0])           # unseen pairs first
    while len(picked) < min(a.n_examples, len(errors)):
        for c in cats:
            if by_cat[c] and len(picked) < a.n_examples:
                picked.append((c, by_cat[c].pop(0)[1]))

    lines = [f"# Error analysis: {stem}\n", f"{len(errors)} label errors / {len(preds)} records.\n",
             "| category | count |\n|---|---|"] + [f"| {k} | {v} |" for k, v in cat_counts.most_common()]
    lines += [f"\nCorrect label but different evidence IDs: {len(wrong_ev)}\n", "## Representative errors\n"]
    for n, (c, e) in enumerate(picked, 1):
        sents = parse_note(data[e["annotation_id"]]["note"])
        lines += [f"### Error {n} - {c}",
                  f"- annotation {e['annotation_id']} (patient {e['patient_id']}), {e['criterion_type']} criterion",
                  f"- criterion: {e['criterion_text']}",
                  f"- expert: **{e['gold_label']}**, evidence {e['gold_ids']}: {ev_text(sents, e['gold_ids'])}",
                  f"- model: **{e['pred_label']}**, evidence {e['pred_ids']}: {ev_text(sents, e['pred_ids'])}",
                  "- your analysis: (write why - missing info? ambiguous criterion? wrong sentence?)\n"]
    Path(f"results/error_analysis_{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote results/error_analysis_{stem}.md and results/confusion_{stem}.png")


if __name__ == "__main__":
    main()