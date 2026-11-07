"""Shared helpers: label sets, note parsing, prompt building, output parsing."""
import json
import re

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EOS_TEXT = "<|im_end|>"

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = sorted({l for v in LABELS.values() for l in v}, key=len, reverse=True)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- patient notes
def parse_note(note: str) -> dict:
    """'0. text\\n1. text' -> {0: 'text', 1: 'text'} (keeps the dataset's sentence IDs)."""
    sents, last = {}, None
    for line in note.split("\n"):
        m = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
        if m:
            last = int(m.group(1))
            sents[last] = m.group(2).strip()
        elif last is not None and line.strip():
            sents[last] += " " + line.strip()
    return sents


def note_to_str(sents: dict) -> str:
    return "\n".join(f"{i}. {s}" for i, s in sorted(sents.items()))


def number_note(text: str) -> dict:
    """For free text typed in the app: keep '0. ...' numbering if present, otherwise split into
    sentences and number them from 0. Returns {sentence_id: sentence}."""
    text = text.strip()
    if re.match(r"^\s*0\.\s", text):
        sents = parse_note(text)
        if sents:
            return sents
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return {i: p for i, p in enumerate(parts)}


# ---------------------------------------------------------------------- prompts
def build_user_prompt(note_str: str, criterion_type: str, criterion_text: str) -> str:
    return (
        "You are screening a patient for a clinical trial.\n\n"
        f"Patient note (sentences are numbered):\n{note_str}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        'Answer in JSON with keys "label" and "evidence_sentences" '
        "(a list of sentence numbers from the note)."
    )


def to_chat(tok, user_prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True
    )


def fit_prompt(tok, sents: dict, ctype: str, ctext: str, max_prompt_tokens: int = 1536):
    """Build the chat prompt. If it is too long, drop sentences from the END of the note
    (and report which IDs were dropped so truncation can be documented)."""
    sents, dropped = dict(sents), []
    while True:
        chat = to_chat(tok, build_user_prompt(note_to_str(sents), ctype, ctext))
        n = len(tok(chat, add_special_tokens=False).input_ids)
        if n <= max_prompt_tokens or len(sents) <= 1:
            return chat, dropped
        last = max(sents)
        sents.pop(last)
        dropped.append(last)


def build_target(label: str, evidence: list) -> str:
    return json.dumps({"label": label, "evidence_sentences": list(evidence)})


# ------------------------------------------------------------- output parsing
def parse_output(text: str, ctype: str = None) -> dict:
    """Tolerant parser. Returns {'label', 'evidence', 'ok'}; ok=False when the JSON was
    broken or the label is not valid for the criterion type."""
    out = {"label": None, "evidence": [], "ok": False}
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            ev = obj.get("evidence_sentences", [])
            ev = sorted({int(i) for i in ev}) if isinstance(ev, list) else []
            out.update(label=str(obj.get("label", "")).strip().lower(), evidence=ev, ok=True)
        except Exception:
            pass
    if not out["ok"]:  # fallback: look for a label string in the raw text
        low = text.lower()
        for lab in ALL_LABELS:
            if lab in low:
                out["label"] = lab
                break
    if ctype is not None and out["label"] not in LABELS[ctype]:
        out["ok"] = False
    return out
