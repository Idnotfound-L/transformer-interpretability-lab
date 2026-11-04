import pandas as pd

from interpretability_lab.sequence_copy import (
    split_disjoint_token_pools,
    make_sequence_copy_cases,
    choose_layer_indices,
    build_component_candidates,
    summarize_interventions,
)


def test_disjoint_pools_and_sequence_copy_layout_are_reproducible():
    tokens = list(range(100, 600))
    discovery_pool, evaluation_pool = split_disjoint_token_pools(tokens, seed=11)
    assert set(discovery_pool).isdisjoint(evaluation_pool)

    first = make_sequence_copy_cases(discovery_pool, 20, seed=23)
    second = make_sequence_copy_cases(discovery_pool, 20, seed=23)
    assert first == second
    all_used = [token for case in first for token in case.input_ids[:-1]] + [case.corruption_token_id for case in first]
    assert len(all_used) == len(set(all_used))
    for case in first:
        assert len(case.input_ids) == 9
        assert case.input_ids[case.first_occurrence_position] == case.repeated_token_id
        assert case.input_ids[case.query_position] == case.repeated_token_id
        assert case.expected_token_id == case.input_ids[case.successor_position]
        assert len(set(case.input_ids[:-1])) == 8
        assert case.corruption_token_id not in case.input_ids


def test_layer_selection_spans_full_depth():
    assert choose_layer_indices(12, 4) == [0, 4, 7, 11]
    assert choose_layer_indices(4, 10) == [0, 1, 2, 3]
    assert choose_layer_indices(12, 1) == [6]


def test_candidates_include_same_layer_random_controls():
    attention = pd.DataFrame([
        {"layer": layer, "head": head, "attention_lift": float(head)}
        for layer in [0, 2] for head in range(4)
    ])
    neurons = {
        0: [(5, 2.0), (6, 1.0)],
        2: [(8, 3.0), (9, 2.0)],
    }
    candidates = build_component_candidates(attention, neurons, [0, 2], 4, 16, 1, 3)
    for (kind, layer, pair_id), group in candidates.groupby(["kind", "layer", "pair_id"]):
        assert set(group.selection) == {"discovery_top", "random_control"}
        assert group.component.nunique() == 2


def test_paired_candidate_control_statistics_detect_positive_effect():
    rows = []
    for prompt_id in range(40):
        noise = (prompt_id % 5) * 0.01
        for selection, drop in [("discovery_top", 0.8 + noise), ("random_control", 0.1 + noise)]:
            rows.append({
                "prompt_id": prompt_id, "kind": "head", "layer": 3,
                "component": 2 if selection == "discovery_top" else 4,
                "selection": selection, "pair_id": "head_L3_R1",
                "copy_margin_drop": drop,
                "kl_baseline_to_intervened": drop,
                "intervened_correct": False,
                "baseline_correct": True,
            })
    summary, paired = summarize_interventions(pd.DataFrame(rows), seed=1)
    assert len(summary) == 2
    assert len(paired) == 1
    assert paired.iloc[0].selected_minus_random_mean_margin_drop > 0.6
    assert paired.iloc[0].ci95_low > 0.6
    assert paired.iloc[0].q_bh <= 0.05


def test_baseline_and_patch_summaries_include_bootstrap_intervals():
    from interpretability_lab.sequence_copy import summarize_baseline_metrics, summarize_activation_patching

    baseline = pd.DataFrame({
        "baseline_correct": [True, False, True, True, False, True],
        "baseline_target_probability": [0.2, 0.01, 0.4, 0.3, 0.02, 0.5],
        "baseline_copy_margin": [1.2, -0.8, 2.0, 1.5, -0.6, 2.1],
    })
    base_summary = summarize_baseline_metrics(baseline, seed=5)
    assert set(base_summary.metric) == {"copy_accuracy", "target_probability", "copy_margin"}
    assert (base_summary.ci95_low <= base_summary.ci95_high).all()

    patch = pd.DataFrame([
        {"layer": layer, "patch_location": location,
         "margin_recovery": 0.2 + layer * 0.01,
         "target_logit_recovery": 0.3 + layer * 0.01}
        for layer in range(3) for location in ["corrupted_successor", "final_query"]
        for _ in range(8)
    ])
    patch_summary = summarize_activation_patching(patch, seed=5)
    assert len(patch_summary) == 6
    assert "margin_recovery_ci95_low" in patch_summary
    assert (patch_summary.margin_recovery_ci95_low <= patch_summary.margin_recovery_ci95_high).all()
