"""Data preparation: load TrialGPT-Criterion-Annotations, drop GPT-4 columns,
make the patient-grouped 37/8/8 split (seed 42), save manifest + jsonl files.
Run from the repo root:  python src/data_prep.py"""
import ast
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

BANNED = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility"]
OUT_DIR = Path("data/processed")


def parse_ids(x):
    if isinstance(x, list):
        return [int(i) for i in x]
    if x is None or str(x).strip() == "":
        return []
    return [int(i) for i in ast.literal_eval(str(x))]


def main():
    df = load_dataset("ncbi/TrialGPT-Criterion-Annotations")["train"].to_pandas()
    df = df.drop(columns=BANNED + ["training"])
    assert not any(c in df.columns for c in BANNED)

    ids = np.array(sorted(df["patient_id"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    train_ids, val_ids, test_ids = ids[:37], ids[37:45], ids[45:]
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "LEAK!"
    assert (len(s_tr), len(s_va), len(s_te)) == (37, 8, len(ids) - 45)

    df["evidence_ids"] = df["expert_sentences"].apply(parse_ids)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": 42, "rule": "sorted unique patient_id, default_rng(42).shuffle, 37/8/8"}
    for name, pset in {"train": s_tr, "val": s_va, "test": s_te}.items():
        part = df[df["patient_id"].isin(pset)]
        part.to_json(OUT_DIR / f"{name}.jsonl", orient="records", lines=True)
        manifest[name] = {"n_patients": len(pset), "n_annotations": int(len(part)),
                          "patient_ids": sorted(map(str, pset))}
        print(name, len(pset), "patients", len(part), "annotations")
    Path("data/split_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
