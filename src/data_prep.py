"""Data preparation: load TrialGPT-Criterion-Annotations, drop GPT-4 columns,
make the patient-grouped 37/8/8 split (seed 42), save manifest + jsonl files."""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

BANNED = ["gpt4_explanation", "explanation_correctness",
          "gpt4_sentences", "gpt4_eligibility"]   # must never be used as input
OUT_DIR = Path("data/processed")


def parse_note(note: str) -> dict:
    """'0. text\\n1. text' -> {0: 'text', 1: 'text'} (keeps original IDs)."""
    sents, last = {}, None
    for line in note.split("\n"):
        m = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
        if m:
            last = int(m.group(1))
            sents[last] = m.group(2).strip()
        elif last is not None and line.strip():   # continuation line
            sents[last] += " " + line.strip()
    return sents


def parse_ids(x) -> list:
    if isinstance(x, list):
        return [int(i) for i in x]
    if x is None or str(x).strip() == "":
        return []
    return [int(i) for i in ast.literal_eval(str(x))]


def build_prompt(note: str, criterion_type: str, criterion_text: str) -> str:
    return (
        "You are screening a patient for a clinical trial.\n\n"
        f"Patient note (sentences are numbered):\n{note}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        'Answer in JSON with keys "label" and "evidence_sentences" '
        "(a list of sentence numbers from the note)."
    )


def build_target(label: str, evidence: list) -> str:
    return json.dumps({"label": label, "evidence_sentences": evidence})


def main():
    ds = load_dataset("ncbi/TrialGPT-Criterion-Annotations")
    df = ds["train"].to_pandas()
    print("Rows:", len(df))

    # --- Drop banned columns + the authors' own 'training' flag ---
    df = df.drop(columns=BANNED + ["training"])
    assert not any(c in df.columns for c in BANNED)

    # --- Patient-grouped split, exactly as the assignment says ---
    ids = np.array(sorted(df["patient_id"].unique()))
    print("Unique patients:", len(ids))
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    train_ids, val_ids, test_ids = ids[:37], ids[37:45], ids[45:]

    # --- Leakage checks ---
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "LEAK!"
    assert len(s_tr) == 37 and len(s_va) == 8 and len(s_te) == len(ids) - 45
    print("Patients -> train/val/test:", len(s_tr), len(s_va), len(s_te))

    # --- Processed records ---
    df["evidence_ids"] = df["expert_sentences"].apply(parse_ids)
    df["prompt"] = [build_prompt(n, t, c) for n, t, c in
                    zip(df["note"], df["criterion_type"], df["criterion_text"])]
    df["target"] = [build_target(l, e) for l, e in
                    zip(df["expert_eligibility"], df["evidence_ids"])]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": s_tr, "val": s_va, "test": s_te}
    manifest = {"seed": 42, "rule": "sorted unique patient_id, rng.shuffle, 37/8/8"}
    for name, pset in splits.items():
        part = df[df["patient_id"].isin(pset)]
        part.to_json(OUT_DIR / f"{name}.jsonl", orient="records", lines=True)
        manifest[name] = {"n_patients": len(pset),
                          "n_annotations": int(len(part)),
                          "patient_ids": sorted(map(str, pset))}
        print(f"\n== {name}: {len(part)} annotations ==")
        print(part.groupby(["criterion_type", "expert_eligibility"]).size())
        print("empty-evidence records:", int((part["evidence_ids"].str.len() == 0).sum()))

    Path("data/split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nSaved data/split_manifest.json and data/processed/*.jsonl")


if __name__ == "__main__":
    main()