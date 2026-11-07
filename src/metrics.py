"""Classification + evidence metrics (overall, inclusion-only, exclusion-only)."""
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def clf_metrics(gold, pred):
    """Macro-F1 is computed over classes that occur in gold OR predictions
    (a class that never occurs anywhere would otherwise add a meaningless 0)."""
    labels = sorted(set(gold) | set(pred))
    rep = classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "per_class": {l: rep[l] for l in labels},
        "labels": labels,
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
    }


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evidence_metrics(gold_lists, pred_lists):
    """EMPTY-SET RULES (state these in the report):
      * gold empty and pred empty  -> P = R = F1 = 1  (correctly cited nothing)
      * exactly one of them empty  -> P = R = F1 = 0
      * otherwise standard set P/R/F1 on sentence IDs.
    'macro_all'      : mean of per-record scores over all records (rules above)
    'macro_nonempty' : mean over records whose gold evidence is non-empty
    'micro'          : pooled TP/FP/FN over all records (empty/empty adds nothing)"""
    per, TP, FP, FN = [], 0, 0, 0
    for g, p in zip(gold_lists, pred_lists):
        g, p = set(g), set(p)
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        if not g and not p:
            s = (1.0, 1.0, 1.0)
        elif not g or not p:
            s = (0.0, 0.0, 0.0)
        else:
            s = _prf(tp, fp, fn)
        per.append((s, bool(g)))

    def mean(xs):
        return [float(np.mean([x[i] for x in xs])) if xs else 0.0 for i in range(3)]

    ne = [s for s, nonempty in per if nonempty]
    return {
        "n": len(per), "n_gold_nonempty": len(ne), "n_gold_empty": len(per) - len(ne),
        "macro_all": dict(zip("PRF", mean([s for s, _ in per]))),
        "macro_nonempty": dict(zip("PRF", mean(ne))),
        "micro": dict(zip("PRF", _prf(TP, FP, FN))),
    }


def compute_all(records):
    """records: dicts with criterion_type, gold_label, pred_label, gold_ids, pred_ids, parse_ok."""
    out = {}
    for name, sub in [("overall", records),
                      ("inclusion", [r for r in records if r["criterion_type"] == "inclusion"]),
                      ("exclusion", [r for r in records if r["criterion_type"] == "exclusion"])]:
        if not sub:
            continue
        out[name] = {
            "classification": clf_metrics([r["gold_label"] for r in sub], [r["pred_label"] for r in sub]),
            "evidence": evidence_metrics([r["gold_ids"] for r in sub], [r["pred_ids"] for r in sub]),
        }
    out["parse_failures"] = int(sum(1 for r in records if not r.get("parse_ok", True)))
    out["pred_label_counts"] = dict(Counter(r["pred_label"] for r in records))
    return out


def print_summary(m, title=""):
    print(f"\n===== {title} =====")
    for k in ("overall", "inclusion", "exclusion"):
        if k not in m:
            continue
        c, e = m[k]["classification"], m[k]["evidence"]
        print(f"[{k}] n={c['n']} acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f} | "
              f"evidence micro F1={e['micro']['F']:.3f} macro(all) F1={e['macro_all']['F']:.3f} "
              f"macro(non-empty gold) F1={e['macro_nonempty']['F']:.3f}")
    print("parse failures:", m["parse_failures"], "| predicted labels:", m["pred_label_counts"])
