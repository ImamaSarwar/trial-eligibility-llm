"""Task 6: browser app.  Start:  python app/app.py   then open http://127.0.0.1:7860
Environment variables (all optional):
  ADAPTER     path or Hugging Face repo id of the LoRA adapter  (empty = base model only)
  BASE_MODEL  default Qwen/Qwen2.5-1.5B-Instruct       DRAFT_MODEL default Qwen/Qwen2.5-0.5B-Instruct
"""
import datetime
import glob
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import BASE_MODEL, DRAFT_MODEL, number_note
from src.pipeline import MODES, Screener

DISCLAIMER = ("**Educational screening prototype - NOT a clinical enrollment decision.** "
              "Outputs may be wrong and must never be used to decide patient care or trial enrollment.")
MAX_CHARS, MAX_CRITERIA = 6000, 12

EXAMPLE_NOTE = ("0. A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest "
                "pain that began two days earlier for the first time in her life.\n1. The pain started while she was walking, "
                "radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea.\n2. The latest episode of "
                "pain ended half an hour prior to her arrival.\n3. She is known to have hypertension and obesity.\n"
                "4. She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.\n"
                "5. She currently takes no medications.\n6. Physical examination is normal.\n7. The EKG shows nonspecific changes.")
EXAMPLE_INC = "Patient is an adult woman or man\nPatient has chest pain\nPatient has a documented history of coronary artery disease"
EXAMPLE_EXC = "Patient currently smokes\nPatient has diabetes mellitus"

_screener = None


def get_screener():
    global _screener
    if _screener is None:
        _screener = Screener(base=os.environ.get("BASE_MODEL", BASE_MODEL),
                             adapter=os.environ.get("ADAPTER") or None,
                             draft=os.environ.get("DRAFT_MODEL", DRAFT_MODEL))
    return _screener


def parse_criteria(text):
    out = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip()
        if line:
            out.append(line)
    return out


def build_report_html(res, note):
    rows = "".join(
        f"<tr><td>{r['type']}</td><td>{html.escape(r['criterion'])}</td><td><b>{html.escape(r['label'])}</b></td>"
        f"<td>{r['sentence_ids']}</td><td>{html.escape(r['evidence_text'])}</td><td>{html.escape(r['explanation'])}</td></tr>"
        for r in res["results"])
    sents = "".join(f"<li>[{i}] {html.escape(s)}</li>" for i, s in res["sentences"].items())
    mi = res["model_info"]
    return f"""<html><head><meta charset="utf-8"><title>Eligibility screening report</title>
<style>body{{font-family:Arial;margin:30px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #999;padding:6px;vertical-align:top}}
th{{background:#eee}}.warn{{background:#fff3cd;padding:10px;border:1px solid #e0c36a}}</style></head><body>
<h1>Clinical-trial eligibility screening report</h1>
<p class="warn"><b>Educational screening prototype - not a clinical enrollment decision.</b></p>
<p><b>Generated:</b> {datetime.datetime.now():%Y-%m-%d %H:%M:%S} &nbsp; <b>Decoding mode:</b> {res['mode']}
&nbsp; <b>Total time:</b> {res['total_seconds']} s</p>
<p><b>Model:</b> {html.escape(mi['base'])} &nbsp; <b>Adapter:</b> {html.escape(str(mi['adapter']))} &nbsp;
<b>Draft:</b> {html.escape(str(mi['draft']))} &nbsp; <b>Device:</b> {mi['device']} {mi['dtype']}</p>
<h2>Preliminary outcome (rule-based aggregation, not a labelled ground truth)</h2><p><b>{html.escape(res['overall'])}</b></p>
<h2>Criterion-level results</h2>
<table><tr><th>Type</th><th>Criterion</th><th>Label</th><th>Sentence IDs</th><th>Evidence</th><th>Explanation</th></tr>{rows}</table>
<h2>Patient note (numbered sentences)</h2><ol start="0" style="list-style:none">{sents}</ol></body></html>"""


def run_screening(note, inclusion, exclusion, mode, k, progress=gr.Progress()):
    note = (note or "").strip()
    inc, exc = parse_criteria(inclusion), parse_criteria(exclusion)
    if not note:
        raise gr.Error("Please paste a patient description.")
    if len(note) > MAX_CHARS:
        raise gr.Error(f"Patient note is too long ({len(note)} characters; maximum {MAX_CHARS}).")
    if len(number_note(note)) < 1:
        raise gr.Error("Could not find any sentences in the patient note.")
    if not inc and not exc:
        raise gr.Error("Please enter at least one inclusion or exclusion criterion (one per line).")
    if len(inc) + len(exc) > MAX_CRITERIA:
        raise gr.Error(f"Too many criteria (maximum {MAX_CRITERIA} per run on this machine).")
    try:
        progress(0, desc="Loading model (first run only)...")
        sc = get_screener()
        res = sc.screen(note, inc, exc, mode=mode, k=int(k), progress=lambda f, d: progress(f, desc=d))
    except gr.Error:
        raise
    except Exception as e:                       # show a readable message instead of a crash
        raise gr.Error(f"Inference failed: {type(e).__name__}: {e}")
    df = pd.DataFrame([{"Type": r["type"], "Criterion": r["criterion"], "Label": r["label"],
                        "Sentence IDs": str(r["sentence_ids"]), "Evidence": r["evidence_text"],
                        "Explanation": r["explanation"], "Seconds": r["seconds"]} for r in res["results"]])
    missing = [r["criterion"] for r in res["results"] if r["label"] == "not enough information"]
    md = f"### Preliminary outcome: {res['overall']}\n\n*Rule-based aggregation of criterion-level outputs - not a ground-truth label.*\n\n"
    if missing:
        md += "**Missing information for:**\n" + "\n".join(f"- {m}" for m in missing) + "\n\n"
    md += f"Mode `{res['mode']}` - {res['total_seconds']} s total - {DISCLAIMER}"
    os.makedirs("results/reports", exist_ok=True)
    path = f"results/reports/report_{datetime.datetime.now():%Y%m%d_%H%M%S}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report_html(res, note))
    return md, df, path


def run_live_benchmark(note, ctype, ctext, n_new, k):
    if not (note or "").strip() or not (ctext or "").strip():
        raise gr.Error("Enter a patient note and one criterion first (use the Screening tab inputs).")
    try:
        rows = get_screener().compare_modes(note, ctype, ctext.strip().splitlines()[0], int(n_new), int(k))
    except Exception as e:
        raise gr.Error(f"Benchmark failed: {type(e).__name__}: {e}")
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(df["mode"], df["total_s"], color=["#888", "#2a7", "#37c"][: len(df)]); ax[0].set_ylabel("total seconds")
    ax[0].set_title("Latency (lower is better)")
    ax[1].bar(df["mode"], df["tokens_per_s"], color=["#888", "#2a7", "#37c"][: len(df)]); ax[1].set_ylabel("tokens / s")
    ax[1].set_title("Throughput (higher is better)")
    plt.tight_layout()
    ok = df["same_tokens_as_no_cache"].all()
    return df, fig, ("All decoders produced identical tokens." if ok else "WARNING: outputs differ between decoders.")


def load_logged():
    files = sorted(glob.glob("results/benchmark_*.json"))
    if not files:
        raise gr.Error("No logged benchmark found in results/. Run `python -m src.benchmark` first (e.g. on Colab).")
    frames, notes = [], []
    for f in files:
        d = json.load(open(f))
        for r in d["results"]:
            frames.append({"file": os.path.basename(f), "device": d["env"]["gpu"], "prompt_len": r["prompt_len"],
                           "new": r["new_tokens"], "mode": r["mode"], "total_s": round(r["total_s"][0], 3),
                           "tok/s": round(r["tokens_per_s"], 1), "peak_MB": None if r["peak_MB"] is None else round(r["peak_MB"]),
                           "identical": r["identical_to_no_cache"], "accept_rate": r.get("acceptance_rate")})
    df = pd.DataFrame(frames)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sub = df[df["file"] == df["file"].iloc[0]]
    for m, g in sub.groupby("mode"):
        ax.plot(g["prompt_len"], g["total_s"], "o-", label=m)
    ax.set_xlabel("prompt length (tokens)"); ax.set_ylabel("total seconds"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    return df, fig


with gr.Blocks(title="Clinical-trial eligibility screening (educational prototype)") as demo:
    gr.Markdown("# Clinical-trial eligibility screening\n" + DISCLAIMER)
    with gr.Tab("Screening"):
        note = gr.Textbox(label="Patient description (plain text, or sentences numbered '0. ...')", lines=10, value=EXAMPLE_NOTE)
        with gr.Row():
            inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=5, value=EXAMPLE_INC)
            exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=5, value=EXAMPLE_EXC)
        with gr.Row():
            mode = gr.Radio(MODES, value="kv_cache", label="Decoding mode")
            k = gr.Slider(1, 8, value=4, step=1, label="Draft tokens k (speculative only)")
        btn = gr.Button("Run screening", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Criterion-level results", wrap=True)
        report = gr.File(label="Download report (HTML)", interactive=False)
        btn.click(run_screening, [note, inc, exc, mode, k], [summary, table, report])
    with gr.Tab("Inference benchmark"):
        gr.Markdown("**Live comparison** on the note and the first line of the criteria above. Slow on CPU - use few new tokens.")
        with gr.Row():
            btype = gr.Radio(["inclusion", "exclusion"], value="inclusion", label="Criterion type")
            nnew = gr.Slider(4, 32, value=8, step=2, label="New tokens")
            bk = gr.Slider(1, 8, value=4, step=1, label="k")
        bbtn = gr.Button("Run live benchmark")
        btable, bplot, bmsg = gr.Dataframe(), gr.Plot(), gr.Markdown()
        bbtn.click(run_live_benchmark, [note, btype, inc, nnew, bk], [btable, bplot, bmsg])
        gr.Markdown("**Logged benchmark** (from `results/benchmark_*.json`, e.g. produced on a Colab GPU)")
        lbtn = gr.Button("Load logged results")
        ltable, lplot = gr.Dataframe(), gr.Plot()
        lbtn.click(load_logged, None, [ltable, lplot])
    with gr.Tab("About"):
        gr.Markdown("Model: Qwen2.5-1.5B-Instruct + LoRA adapter (`ADAPTER`), draft Qwen2.5-0.5B-Instruct. Evidence text is always "
                    "copied from the note by sentence ID. Explanations are templates built from the model's label and cited sentences.")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
