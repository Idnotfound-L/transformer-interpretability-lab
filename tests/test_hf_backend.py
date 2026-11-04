from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from interpretability_lab.hf_backend import HuggingFaceTransformerAdapter


class FakeAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.o_proj(x)


class FakeMLP(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.up_proj = nn.Linear(width, intermediate)
        self.down_proj = nn.Linear(intermediate, width)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


class FakeBlock(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.self_attn = FakeAttention(width)
        self.mlp = FakeMLP(width, intermediate)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        x = x + self.self_attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return (x,)


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        values = [ord(c) % 31 for c in text]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr((int(i) % 26) + 97) for i in ids)


class FakeHFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=12,
            num_hidden_layers=2,
        )
        self.embed = nn.Embedding(256, 8)
        self.layers = nn.ModuleList([FakeBlock(8, 12), FakeBlock(8, 12)])
        self.lm_head = nn.Linear(8, 256)

    def forward(self, input_ids, use_cache=False, output_attentions=False, return_dict=True):
        x = self.embed(input_ids)
        attention_values = []
        batch, seq_len = input_ids.shape
        for layer in self.layers:
            x = layer(x)[0]
            if output_attentions:
                pattern = torch.zeros(batch, 2, seq_len, seq_len, device=input_ids.device)
                for query in range(seq_len):
                    pattern[:, :, query, :query + 1] = 1.0 / (query + 1)
                attention_values.append(pattern)
        return SimpleNamespace(logits=self.lm_head(x), attentions=tuple(attention_values) if output_attentions else None)


def make_adapter():
    torch.manual_seed(17)
    return HuggingFaceTransformerAdapter(FakeHFModel(), FakeTokenizer(), "fake/model")


def test_resolves_gqa_architecture_and_projection_widths():
    adapter = make_adapter()
    assert adapter.n_layers == 2
    assert adapter.n_heads == 2
    assert adapter.n_kv_heads == 1
    assert adapter.head_dim == 4
    assert adapter.intermediate_size == 12
    assert adapter.layers[0].attention_output_path.endswith("self_attn.o_proj")
    assert adapter.layers[0].mlp_output_path.endswith("mlp.down_proj")


def test_head_and_neuron_ablations_produce_finite_metrics_and_remove_hooks():
    adapter = make_adapter()
    ids = adapter.encode("example prompt")
    baseline_head_hooks = len(adapter.layers[0].attention_output._forward_pre_hooks)
    head_result = adapter.ablation_metrics(ids, "head", 0, 1)
    neuron_result = adapter.ablation_metrics(ids, "neuron", 1, 3)
    assert torch.isfinite(torch.tensor(head_result["kl_baseline_to_intervened"]))
    assert torch.isfinite(torch.tensor(neuron_result["kl_baseline_to_intervened"]))
    assert len(adapter.layers[0].attention_output._forward_pre_hooks) == baseline_head_hooks
    assert len(adapter.layers[1].mlp_output._forward_pre_hooks) == 0


def test_activation_capture_and_patching():
    adapter = make_adapter()
    clean = adapter.encode("alpha")
    corrupt = adapter.encode("bravo")
    captured = adapter.capture_mlp_intermediates(clean, [0, 1])
    assert set(captured) == {0, 1}
    assert captured[0].shape[-1] == adapter.intermediate_size
    rows = adapter.activation_patch_sweep(clean, corrupt, layer_count=2)
    assert len(rows) == 2
    assert all("logit_recovery_fraction" in row for row in rows)
    # Re-inserting the corrupted activation at the same layer should preserve its baseline target logit.
    assert all(abs(row["corrupted_self_patch_control_logit"] - row["corrupted_target_logit"]) < 1e-5 for row in rows)


def test_activation_patch_rejects_misaligned_tokens():
    adapter = make_adapter()
    with pytest.raises(ValueError, match="identical shapes"):
        adapter.activation_patch_sweep(adapter.encode("short"), adapter.encode("longer"))


def test_index_validation():
    adapter = make_adapter()
    with pytest.raises(IndexError):
        with adapter.ablate_head(0, 2):
            pass
    with pytest.raises(IndexError):
        with adapter.ablate_neuron(0, 12):
            pass


def test_huggingface_repeated_token_probe_helper():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_hf_research import attention_probe

    adapter = make_adapter()
    frame = attention_probe(adapter, ["zzzz", "qqqq"], layer_count=2)
    assert not frame.empty
    assert {"layer", "head", "attention_to_matching_occurrence", "match_attention_lift"}.issubset(frame.columns)
    assert set(frame["layer"].unique()) == {0, 1}


def test_position_scoped_ablation_only_targets_selected_sequence_position():
    adapter = make_adapter()
    ids = adapter.encode("abcdefgh")
    layer = adapter.layers[0]
    observed = {}

    def capture(_module, inputs):
        observed["value"] = inputs[0].detach().clone()

    with adapter.ablate_head(0, 1, position=-1):
        handle = layer.attention_output.register_forward_pre_hook(capture)
        adapter.forward(ids)
        handle.remove()
    # The captured tensor is after the intervention hook; the selected head slice
    # is zero at the final position and remains available at earlier positions.
    start, stop = adapter.head_dim, 2 * adapter.head_dim
    assert torch.allclose(observed["value"][:, -1, start:stop], torch.zeros_like(observed["value"][:, -1, start:stop]))
    assert torch.any(observed["value"][:, :-1, start:stop] != 0)


def test_position_scoped_neuron_ablation_rejects_invalid_position():
    adapter = make_adapter()
    ids = adapter.encode("abcdefgh")
    with pytest.raises(IndexError, match="Position"):
        with adapter.ablate_neuron(0, 1, position=10_000):
            adapter.forward(ids)


def test_sequence_copy_study_pipeline_on_synthetic_adapter():
    from interpretability_lab.sequence_copy import (
        make_sequence_copy_cases,
        collect_copy_attention_scores,
        rank_mlp_neurons,
        build_component_candidates,
        evaluate_copy_interventions,
        summarize_interventions,
        run_copy_activation_patching,
    )

    adapter = make_adapter()
    discovery = make_sequence_copy_cases(list(range(1, 51)), 4, seed=1)
    evaluation = make_sequence_copy_cases(list(range(51, 101)), 4, seed=2, start_prompt_id=100)
    layers = [0, 1]
    attention = collect_copy_attention_scores(adapter, discovery, layers, batch_size=2)
    neurons = rank_mlp_neurons(adapter, discovery, layers, batch_size=2)
    candidates = build_component_candidates(
        attention, neurons, layers, adapter.n_heads, adapter.intermediate_size, 1, seed=3
    )
    baseline, details = evaluate_copy_interventions(adapter, evaluation, candidates, batch_size=2)
    summary, paired = summarize_interventions(details, seed=4)
    patching = run_copy_activation_patching(adapter, evaluation[:2], layers)

    assert len(attention) == len(discovery) * len(layers) * adapter.n_heads
    assert len(baseline) == len(evaluation)
    assert not details.empty and not summary.empty and not paired.empty
    assert not patching.empty
    assert set(patching.patch_location) == {"corrupted_successor", "final_query"}
    assert details.copy_margin_drop.map(lambda x: torch.isfinite(torch.tensor(x))).all()


def test_protocol_v2_attention_targets_successor_b_and_retains_first_a_diagnostic():
    from interpretability_lab.sequence_copy_v2 import collect_copy_attention_scores_v2
    from interpretability_lab.sequence_copy import make_sequence_copy_cases

    adapter = make_adapter()
    cases = make_sequence_copy_cases(list(range(1, 51)), 3, seed=33)
    scores = collect_copy_attention_scores_v2(adapter, cases, [0, 1], batch_size=2)
    assert len(scores) == len(cases) * 2 * adapter.n_heads
    assert "attention_lift_to_successor_B" in scores
    assert "attention_lift_to_first_A" in scores
    assert (scores.successor_B_position > scores.first_A_position).all()


def test_protocol_v2_neurons_are_ranked_by_clean_corrupt_delta_and_control_cases_have_no_repeats():
    from interpretability_lab.sequence_copy_v2 import (
        rank_mlp_neurons_v2, make_random_transition_control_cases,
    )
    from interpretability_lab.sequence_copy import make_sequence_copy_cases

    adapter = make_adapter()
    cases = make_sequence_copy_cases(list(range(1, 51)), 3, seed=35)
    ranked = rank_mlp_neurons_v2(adapter, cases, [0, 1], batch_size=2)
    assert set(ranked) == {0, 1}
    assert len(ranked[0][0]) == 3
    assert all(len(item) == 3 for item in ranked[0])
    control = make_random_transition_control_cases(list(range(101, 301)), 8, seed=36)
    for case in control:
        assert len(set(case.input_ids)) == len(case.input_ids)
        assert case.expected_token_id not in case.input_ids


def test_protocol_v2_component_patching_uses_random_controls_and_excludes_trivial_final_query():
    from interpretability_lab.sequence_copy_v2 import (
        collect_copy_attention_scores_v2, rank_mlp_neurons_v2,
        build_component_candidates_v2, run_copy_activation_patching_v2,
        summarize_activation_patching_v2,
    )
    from interpretability_lab.sequence_copy import make_sequence_copy_cases

    adapter = make_adapter()
    discovery = make_sequence_copy_cases(list(range(1, 51)), 4, seed=41)
    evaluation = make_sequence_copy_cases(list(range(51, 101)), 4, seed=42, start_prompt_id=100)
    layers = [0, 1]
    attention = collect_copy_attention_scores_v2(adapter, discovery, layers, batch_size=2)
    neurons = rank_mlp_neurons_v2(adapter, discovery, layers, batch_size=2)
    candidates = build_component_candidates_v2(
        attention, neurons, layers, adapter.n_heads, adapter.intermediate_size, 1, seed=43
    )
    patching = run_copy_activation_patching_v2(adapter, evaluation[:2], layers, candidates, seed=44)
    assert not patching.empty
    assert "final_query" not in set(patching.patch_location)
    assert {"head_context_final_query", "mlp_neuron_final_query", "block_successor_position", "block_random_position"}.issubset(set(patching.patch_location))
    summary = summarize_activation_patching_v2(patching, seed=45)
    assert not summary.empty
    assert "selection" in summary.columns
