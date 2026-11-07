"""Task 5: greedy speculative decoding with explicit KV-cache management for both models.

INVARIANT: with L = len(prompt) + len(generated) committed tokens, both caches hold the first
L-1 tokens; the last committed token ("pending") has not been fed to either model yet.

One round:
  1. DRAFT  : the draft model greedily proposes kk tokens d1..dkk (feeding pending, d1, ...).
  2. VERIFY : the target processes [pending, d1..dkk] in ONE forward pass -> its own greedy choice
              g0..gkk at every position (g_i = what the target emits after input i).
  3. ACCEPT : n_acc = longest prefix with g_i == d_{i+1}. New tokens = d1..d_nacc + g_nacc
              (g_nacc is the correction on a mismatch, or a free bonus token if all matched).
  4. SYNC   : crop target and draft caches back to L_new-1 entries (drops rejected K/V). If every
              draft was accepted the draft cache is one token short, so that token is fed to it.
The output is token-for-token what greedy decoding of the target alone would produce."""
import time

import torch

from .kv_decode import sync


def crop_cache(cache, keep_len):
    """Shrink a DynamicCache so it holds exactly `keep_len` positions. Uses the negative
    'remove n tokens' form of crop(), which works in both old and new transformers versions."""
    extra = cache.get_seq_length() - keep_len
    assert extra >= 0, "cache is shorter than the length we want to keep"
    if extra > 0:
        cache.crop(-extra)


def _fwd(model, ids, start_pos, cache):
    n = ids.shape[1]
    dev = ids.device
    return model(
        input_ids=ids,
        past_key_values=cache,
        position_ids=torch.arange(start_pos, start_pos + n, device=dev).unsqueeze(0),
        attention_mask=torch.ones(1, start_pos + n, dtype=torch.long, device=dev),
        use_cache=True,
    )


@torch.no_grad()
def speculative_generate(target, draft, input_ids, n_new, k=4, eos_id=None):
    dev = input_ids.device
    n = input_ids.shape[1]
    stats = {"rounds": 0, "drafted": 0, "accepted": 0, "target_forwards": 1,
             "draft_forwards": 1, "accepted_per_round": []}

    sync(dev); t0 = time.perf_counter()
    out_t = target(input_ids=input_ids, use_cache=True)                 # target prefill
    tc = out_t.past_key_values
    dc = draft(input_ids=input_ids, use_cache=True).past_key_values     # draft prefill
    pending = out_t.logits[0, -1].argmax().item()
    new = [pending]
    sync(dev); t_first = time.perf_counter()

    while len(new) < n_new and not (eos_id is not None and eos_id in new):
        L = n + len(new)                                  # committed length; pending is index L-1
        kk = max(0, min(k, n_new - len(new) - 1))         # never produce more than n_new tokens
        pend_t = torch.tensor([[pending]], device=dev)

        # 1. draft kk tokens
        drafts, x = [], pend_t
        for i in range(kk):
            o = _fwd(draft, x, L - 1 + i, dc)
            d = o.logits[0, -1].argmax().item()
            drafts.append(d)
            x = torch.tensor([[d]], device=dev)
        stats["draft_forwards"] += kk

        # 2. verify with ONE target pass over [pending, d1..dkk]
        ver = torch.tensor([[pending] + drafts], device=dev)
        o = _fwd(target, ver, L - 1, tc)
        g = o.logits[0].argmax(dim=-1).tolist()           # kk+1 greedy target choices
        stats["target_forwards"] += 1

        # 3. accept the longest matching prefix, then take the target's own token
        n_acc = 0
        while n_acc < kk and g[n_acc] == drafts[n_acc]:
            n_acc += 1
        accepted = drafts[:n_acc] + [g[n_acc]]
        new.extend(accepted)
        stats["rounds"] += 1
        stats["drafted"] += kk
        stats["accepted"] += n_acc
        stats["accepted_per_round"].append(n_acc)

        # 4. synchronise caches to L_new - 1 = L + n_acc entries
        crop_cache(tc, L + n_acc)
        if n_acc == kk:                                   # draft cache is one token short
            missing = drafts[-1] if kk > 0 else pending
            _fwd(draft, torch.tensor([[missing]], device=dev), L - 1 + kk, dc)
            stats["draft_forwards"] += 1
        else:
            crop_cache(dc, L + n_acc)
        pending = accepted[-1]

    if eos_id is not None and eos_id in new:
        new = new[: new.index(eos_id) + 1]
    new = new[:n_new]
    sync(dev); t_end = time.perf_counter()
    stats["acceptance_rate"] = stats["accepted"] / stats["drafted"] if stats["drafted"] else 0.0
    stats["tokens_per_target_pass"] = len(new) / stats["target_forwards"]
    return {"tokens": new, "prefill_s": t_first - t0, "decode_s": t_end - t_first, "stats": stats}
