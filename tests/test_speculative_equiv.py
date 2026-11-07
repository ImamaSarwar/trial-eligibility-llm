import pytest
import torch
from src.kv_decode import generate_with_cache
from src.speculative import speculative_generate


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("prompt_len,n_new", [(16, 20), (64, 33), (150, 40)])
def test_equals_greedy_with_noisy_draft(tiny_target, noisy_draft, k, prompt_len, n_new):
    torch.manual_seed(prompt_len + k)
    ids = torch.randint(0, 200, (1, prompt_len))
    ref = generate_with_cache(tiny_target, ids, n_new)["tokens"]
    res = speculative_generate(tiny_target, noisy_draft, ids, n_new, k=k)
    assert res["tokens"] == ref
    assert len(res["tokens"]) == n_new


def test_perfect_draft_accepts_everything(tiny_target):
    torch.manual_seed(3)
    ids = torch.randint(0, 200, (1, 30))
    ref = generate_with_cache(tiny_target, ids, 25)["tokens"]
    res = speculative_generate(tiny_target, tiny_target, ids, 25, k=4)
    assert res["tokens"] == ref
    assert res["stats"]["acceptance_rate"] > 0.95
    assert res["stats"]["tokens_per_target_pass"] > 2


def test_eos_stops(tiny_target, noisy_draft):
    torch.manual_seed(7)
    ids = torch.randint(0, 200, (1, 20))
    ref = generate_with_cache(tiny_target, ids, 30)["tokens"]
    eos = ref[6]
    res = speculative_generate(tiny_target, noisy_draft, ids, 30, k=4, eos_id=eos)
    assert res["tokens"] == ref[: ref.index(eos) + 1]
