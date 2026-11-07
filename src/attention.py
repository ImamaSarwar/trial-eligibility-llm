"""Task 4: reference scaled dot-product attention and tiled attention with online softmax.
Shapes: q (B,H,Nq,D), k/v (B,H,Nk,D). Causal masking uses 'bottom-right' alignment: query i may
see keys j <= i + (Nk - Nq). This also covers decoding with a KV cache (Nq < Nk)."""
import math

import torch


def reference_attention(q, k, v, causal=False):
    """softmax(Q K^T / sqrt(d)) V, materialising the full Nq x Nk score matrix.
    Numerically stable: the row maximum is subtracted before exp()."""
    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    nq, nk = q.shape[-2], k.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        assert nk >= nq, "causal attention needs Nk >= Nq"
        allowed = torch.ones(nq, nk, dtype=torch.bool, device=q.device).tril(diagonal=nk - nq)
        scores = scores.masked_fill(~allowed, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)      # stability
    w = torch.exp(scores)
    w = w / w.sum(dim=-1, keepdim=True)
    return (w @ v).to(dtype)


def tiled_attention(q, k, v, causal=False, block_q=64, block_k=64):
    """FlashAttention-style algorithm in plain PyTorch. For every block of queries we sweep over
    blocks of keys, keeping per query row a running max m, running normaliser l and a running
    (un-normalised) output o. The Nq x Nk matrix is never built; only block_q x block_k tiles."""
    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    B, H, nq, D = q.shape
    nk = k.shape[-2]
    assert not causal or nk >= nq, "causal attention needs Nk >= Nq"
    scale, offset = 1.0 / math.sqrt(D), nk - nq
    out = torch.empty_like(q)

    for qs in range(0, nq, block_q):
        qe = min(qs + block_q, nq)
        qb = q[:, :, qs:qe]
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        o = torch.zeros((B, H, qe - qs, D), device=q.device)
        for ks in range(0, nk, block_k):
            if causal and ks > (qe - 1) + offset:      # whole key block lies in the future
                break
            ke = min(ks + block_k, nk)
            s = (qb @ k[:, :, ks:ke].transpose(-2, -1)) * scale
            if causal:
                qi = torch.arange(qs, qe, device=q.device)[:, None] + offset
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_safe)               # rescales everything accumulated so far
            p = torch.exp(s - m_safe)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            o = o * alpha + p @ v[:, :, ks:ke]
            m = m_new
        out[:, :, qs:qe] = o / l
    return out.to(dtype)
