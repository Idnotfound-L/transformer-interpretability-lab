import torch
from interpretability_lab.core import GPTConfig, TinyGPT
from interpretability_lab.research import (
    make_induction_prompt, induction_attention_score, activation_patch_sweep,
    paired_summary,
)


def test_induction_prompt_and_score_shape():
    prompt = make_induction_prompt(1, 2, [3, 4, 5])
    assert prompt.tolist() == [[1, 2, 3, 4, 5, 1]]
    attn = torch.zeros(1, 2, 6, 6)
    attn[0, 0, 5, 0] = 0.7
    attn[0, 1, 5, 0] = 0.3
    score = induction_attention_score(attn, 0, 5)
    assert score.shape == (2,)
    assert torch.allclose(score, torch.tensor([0.7, 0.3]))


def test_activation_patch_returns_layerwise_results():
    torch.manual_seed(0)
    model = TinyGPT(GPTConfig(vocab_size=12, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0))
    clean = torch.tensor([[1, 2, 3, 4]])
    corrupt = torch.tensor([[1, 5, 3, 4]])
    result = activation_patch_sweep(model, clean, corrupt, target_token=6)
    assert len(result["patched_target_logits"]) == 2
    assert len(result["recovery_fraction"]) == 2


def test_paired_summary_has_bootstrap_interval():
    result = paired_summary([1.0, 2.0, 3.0], bootstrap_samples=100)
    assert result["n"] == 3
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]
