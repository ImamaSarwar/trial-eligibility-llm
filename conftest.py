"""pytest fixtures: tiny random Qwen2-style models (CPU, fp32) so tests run in seconds without
downloading anything. They check ALGORITHM correctness, not language quality."""
import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def _tiny(seed, layers):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def tiny_target():
    return _tiny(0, 4)


@pytest.fixture(scope="session")
def noisy_draft(tiny_target):
    d = copy.deepcopy(tiny_target)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in d.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return d
