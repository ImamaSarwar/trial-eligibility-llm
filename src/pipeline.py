"""One interface for the app: patient note + criteria -> structured, evidence-linked results.
Decoding modes: 'no_cache' | 'kv_cache' | 'speculative' (all hand-written, all greedy)."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import BASE_MODEL, DRAFT_MODEL, EOS_TEXT, fit_prompt, number_note, parse_output
from .kv_decode import generate_no_cache, generate_with_cache
from .speculative import speculative_generate

MODES = ["kv_cache", "speculative", "no_cache"]


def aggregate(results):
    """EXPLICIT RULES (an aggregation of criterion-level outputs, NOT a ground-truth label):
    1 any unparsable/invalid prediction      -> needs manual review
    2 any exclusion criterion 'excluded'    -> likely ineligible
    3 any inclusion criterion 'not included'-> likely ineligible
    4 any 'not enough information'          -> needs review (information missing)
    5 otherwise                             -> likely eligible (preliminary).  'not applicable' is ignored."""
    labels = [(r["type"], r["label"]) for r in results]
    if any(l == "invalid" for _, l in labels):
        return "NEEDS REVIEW - at least one prediction could not be parsed"
    if any(t == "exclusion" and l == "excluded" for t, l in labels):
        return "LIKELY INELIGIBLE - an exclusion criterion appears to be met"
    if any(t == "inclusion" and l == "not included" for t, l in labels):
        return "LIKELY INELIGIBLE - an inclusion criterion appears not to be met"
    n = sum(l == "not enough information" for _, l in labels)
    if n:
        return f"NEEDS REVIEW - information is missing for {n} criterion/criteria"
    return "LIKELY ELIGIBLE (preliminary; all applicable criteria satisfied)"


def explain(label, ids, evidence_text):
    """Template explanation built ONLY from the model's label and the cited sentences."""
    if label == "invalid":
        return "The model output could not be parsed; no conclusion is drawn."
    if label == "not enough information":
        base = "The note does not contain enough information to decide this criterion."
        return base + (f" Possibly related sentence(s) {ids}: \"{evidence_text}\"." if ids else " No sentence addresses it.")
    if label == "not applicable":
        return "The criterion was judged not applicable to this patient." + (f" See sentence(s) {ids}." if ids else "")
    if not ids:
        return f"Predicted '{label}' but the model cited no supporting sentence - treat with caution."
    return f"Predicted '{label}' based on sentence(s) {ids}: \"{evidence_text}\""


class Screener:
    def __init__(self, base=BASE_MODEL, adapter=None, draft=DRAFT_MODEL, device=None, load_draft=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=self.dtype)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.draft = None
        if load_draft and draft:
            self.draft = AutoModelForCausalLM.from_pretrained(draft, torch_dtype=self.dtype).to(self.device).eval()
        self.eos_id = self.tok.convert_tokens_to_ids(EOS_TEXT)
        self.info = {"base": base, "adapter": adapter or "none (base model)", "draft": draft if self.draft else "not loaded",
                     "device": self.device, "dtype": str(self.dtype)}

    def _encode(self, sents, ctype, ctext):
        chat, dropped = fit_prompt(self.tok, sents, ctype, ctext)
        ids = self.tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        return ids, dropped

    def _generate(self, ids, mode, n_new=64, k=4):
        if mode == "no_cache":
            return generate_no_cache(self.model, ids, n_new, self.eos_id)
        if mode == "speculative":
            if self.draft is None:
                raise RuntimeError("Draft model not loaded")
            return speculative_generate(self.model, self.draft, ids, n_new, k=k, eos_id=self.eos_id)
        return generate_with_cache(self.model, ids, n_new, self.eos_id)

    def screen(self, note_text, inclusion, exclusion, mode="kv_cache", k=4, progress=None):
        sents = number_note(note_text)
        crits = [("inclusion", c) for c in inclusion] + [("exclusion", c) for c in exclusion]
        results, t_all = [], time.perf_counter()
        for n, (ctype, ctext) in enumerate(crits):
            if progress:
                progress(n / len(crits), f"Criterion {n+1}/{len(crits)}")
            ids, dropped = self._encode(sents, ctype, ctext)
            t = time.perf_counter()
            gen = self._generate(ids, mode, k=k)
            elapsed = time.perf_counter() - t
            text = self.tok.decode(gen["tokens"], skip_special_tokens=True)
            p = parse_output(text, ctype)
            label = p["label"] if p["ok"] else "invalid"
            cited = [i for i in p["evidence"] if i in sents]           # drop hallucinated IDs
            ev_text = " ".join(sents[i] for i in cited)                # text always comes from the note
            results.append({"type": ctype, "criterion": ctext, "label": label, "sentence_ids": cited,
                            "evidence_text": ev_text, "explanation": explain(label, cited, ev_text),
                            "raw_output": text, "seconds": round(elapsed, 2), "n_tokens": len(gen["tokens"]),
                            "prompt_tokens": ids.shape[1], "truncated_sentences": dropped})
        return {"results": results, "overall": aggregate(results), "sentences": sents, "mode": mode,
                "total_seconds": round(time.perf_counter() - t_all, 2), "model_info": self.info}

    def compare_modes(self, note_text, criterion_type, criterion_text, n_new=8, k=4):
        """Live mini-benchmark: same prompt, fixed number of new tokens, all three decoders."""
        ids, _ = self._encode(number_note(note_text), criterion_type, criterion_text)
        rows, ref = [], None
        for mode in ["no_cache", "kv_cache", "speculative"]:
            if mode == "speculative" and self.draft is None:
                continue
            t = time.perf_counter()
            g = self._generate_fixed(ids, mode, n_new, k)
            total = time.perf_counter() - t
            ref = g["tokens"] if ref is None else ref
            row = {"mode": mode, "prompt_tokens": ids.shape[1], "new_tokens": len(g["tokens"]),
                   "prefill_s": round(g["prefill_s"], 3), "decode_s": round(g["decode_s"], 3),
                   "total_s": round(total, 3), "tokens_per_s": round(len(g["tokens"]) / total, 2),
                   "same_tokens_as_no_cache": g["tokens"] == ref}
            if mode == "speculative":
                row["acceptance_rate"] = round(g["stats"]["acceptance_rate"], 3)
            rows.append(row)
        return rows

    def _generate_fixed(self, ids, mode, n_new, k):
        if mode == "no_cache":
            return generate_no_cache(self.model, ids, n_new, None)
        if mode == "speculative":
            return speculative_generate(self.model, self.draft, ids, n_new, k=k, eos_id=None)
        return generate_with_cache(self.model, ids, n_new, None)
