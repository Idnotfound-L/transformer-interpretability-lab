import torch

from interpretability_lab.core import GPTConfig, TinyGPT, CharTokenizer
from interpretability_lab.experiment import bootstrap_mean, seed_everything, sha256_texts
from interpretability_lab.research import sweep_components


def test_saved_tokenizer_vocabulary_can_be_restored():
    original = CharTokenizer("cabba")
    restored = CharTokenizer(chars=original.chars)
    assert restored.encode("cab") == original.encode("cab")
    assert restored.decode(restored.encode("cab")) == "cab"


def test_bootstrap_mean_is_deterministic_and_prompt_level():
    first = bootstrap_mean([0.2, 0.4, 0.8], seed=11, samples=500)
    second = bootstrap_mean([0.2, 0.4, 0.8], seed=11, samples=500)
    assert first == second
    assert first["n"] == 3
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def test_prompt_hash_stable():
    assert sha256_texts(["a", "b"]) == sha256_texts(["a", "b"])
    assert sha256_texts(["a", "b"]) != sha256_texts(["b", "a"])


def test_component_sweep_covers_every_head_and_each_prompt():
    seed_everything(5)
    model = TinyGPT(GPTConfig(10, block_size=8, n_layer=1, n_head=2, n_embd=8, dropout=0.0)).eval()
    prompts = [torch.tensor([[1, 2, 3]]), torch.tensor([[1, 4, 3]])]
    detail, summary = sweep_components(model, prompts, "head", 0)
    assert len(detail) == 4  # 2 heads x 2 prompts
    assert len(summary) == 2
    assert all(row["kl_n"] == 2 for row in summary)
