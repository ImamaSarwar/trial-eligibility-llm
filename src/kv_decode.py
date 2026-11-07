"""Task 3: hand-written greedy decoding WITHOUT and WITH an explicit KV cache (batch size 1)."""
import time

import torch


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def kv_cache_bytes(config, seq_len, bytes_per_el=2):
    """Analytic cache size: 2 (K and V) x layers x kv_heads x head_dim x seq_len x bytes."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * seq_len * bytes_per_el


@torch.no_grad()
def generate_no_cache(model, input_ids, n_new, eos_id=None):
    """Reference path: every step re-runs the WHOLE sequence (use_cache=False)."""
    dev = input_ids.device
    ids, new = input_ids, []
    sync(dev); t0 = time.perf_counter(); t_first = None
    for step in range(n_new):
        logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        new.append(nxt.item())
        if step == 0:
            sync(dev); t_first = time.perf_counter()
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and new[-1] == eos_id:
            break
    sync(dev); t_end = time.perf_counter()
    return {"tokens": new, "prefill_s": t_first - t0, "decode_s": t_end - t_first}


@torch.no_grad()
def generate_with_cache(model, input_ids, n_new, eos_id=None):
    """1) PREFILL: run the prompt once and keep the returned key/value cache.
    2) DECODE: feed ONLY the newest token. We pass the cache, the explicit absolute position of the
       new token (= current cache length, needed by RoPE) and an attention mask covering
       cache + new token. The model appends the new K/V to the cache at every step."""
    dev = input_ids.device
    sync(dev); t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    new = [nxt.item()]
    sync(dev); t_first = time.perf_counter()
    cur_len = input_ids.shape[1]                      # tokens currently stored in the cache
    while len(new) < n_new and not (eos_id is not None and new[-1] == eos_id):
        out = model(
            input_ids=nxt,                                            # (1,1): only the new token
            past_key_values=past,
            position_ids=torch.tensor([[cur_len]], device=dev),       # its absolute position
            attention_mask=torch.ones(1, cur_len + 1, dtype=torch.long, device=dev),
            use_cache=True,
        )
        past = out.past_key_values                    # cache grew by one position
        cur_len += 1
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        new.append(nxt.item())
    sync(dev); t_end = time.perf_counter()
    return {"tokens": new, "prefill_s": t_first - t0, "decode_s": t_end - t_first,
            "cache_len": cur_len}
