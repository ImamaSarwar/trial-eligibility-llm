import pytest
import torch
import torch.nn.functional as F
from src.attention import reference_attention, tiled_attention

SHAPES = [(1, 1, 8, 4), (2, 3, 37, 16), (1, 4, 128, 32), (2, 2, 100, 64), (1, 2, 1, 16)]


def _sdpa(q, k, v, causal):
    nq, nk = q.shape[-2], k.shape[-2]
    mask = torch.ones(nq, nk, dtype=torch.bool).tril(diagonal=nk - nq) if causal else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("blocks", [(16, 16), (7, 13), (64, 64)])
def test_tiled_matches_reference(shape, causal, blocks):
    torch.manual_seed(0)
    B, H, N, D = shape
    q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
    ref = reference_attention(q, k, v, causal)
    til = tiled_attention(q, k, v, causal, *blocks)
    assert (ref - til).abs().max() < 1e-5
    assert (ref - _sdpa(q, k, v, causal)).abs().max() < 1e-5


def test_decode_shape_nq_less_than_nk():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 16); k = torch.randn(1, 2, 50, 16); v = torch.randn(1, 2, 50, 16)
    ref = reference_attention(q, k, v, True)
    assert (ref - tiled_attention(q, k, v, True, 2, 8)).abs().max() < 1e-5
    assert (ref - _sdpa(q, k, v, True)).abs().max() < 1e-5


def test_large_scores_are_stable():
    torch.manual_seed(0)
    q = 100 * torch.randn(1, 2, 64, 16); k = 100 * torch.randn(1, 2, 64, 16); v = torch.randn(1, 2, 64, 16)
    assert torch.isfinite(reference_attention(q, k, v, True)).all()
    assert torch.isfinite(tiled_attention(q, k, v, True, 16, 16)).all()
    assert (reference_attention(q, k, v, True) - tiled_attention(q, k, v, True, 16, 16)).abs().max() < 1e-4


def test_causal_no_future_leak():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 10, 8) for _ in range(3))
    out1 = tiled_attention(q, k, v, True, 4, 4)
    v2 = v.clone(); v2[:, :, 9] += 100.0          # change ONLY the last (future) value
    out2 = tiled_attention(q, k, v2, True, 4, 4)
    assert torch.allclose(out1[:, :, :9], out2[:, :, :9])
