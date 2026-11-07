import pytest
import torch
from src.kv_decode import generate_no_cache, generate_with_cache


@pytest.mark.parametrize("prompt_len,n_new", [(16, 8), (64, 16), (200, 32)])
def test_cached_equals_uncached(tiny_target, prompt_len, n_new):
    torch.manual_seed(prompt_len)
    ids = torch.randint(0, 200, (1, prompt_len))
    a = generate_no_cache(tiny_target, ids, n_new)["tokens"]
    b = generate_with_cache(tiny_target, ids, n_new)["tokens"]
    assert a == b


def test_matches_hf_generate(tiny_target):
    torch.manual_seed(5)
    ids = torch.randint(0, 200, (1, 40))
    ref = tiny_target.generate(ids, max_new_tokens=12, min_new_tokens=12, do_sample=False,
                               pad_token_id=0)[0, 40:].tolist()
    assert generate_with_cache(tiny_target, ids, 12)["tokens"] == ref
