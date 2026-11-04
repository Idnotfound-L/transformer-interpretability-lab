#!/usr/bin/env python3
"""Run Protocol V2 for sequence-copying mechanistic interpretability.

V2 changes: rank attention to the earlier successor B; rank MLP neurons by clean-vs-
corrupted final-query activation sensitivity; use activity-matched neuron controls;
add a repetition-free random-transition control task; and perform component-specific
head/neuron patching plus non-trivial block-position controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from interpretability_lab.experiment import seed_everything, write_json
from interpretability_lab.hf_backend import load_huggingface_model
from interpretability_lab.sequence_copy import (
    lexical_token_ids, make_sequence_copy_cases, choose_layer_indices,
    evaluate_copy_interventions, summarize_interventions, summarize_baseline_metrics,
)
from interpretability_lab.sequence_copy_v2 import (
    split_three_disjoint_token_pools, make_random_transition_control_cases,
    collect_copy_attention_scores_v2, rank_mlp_neurons_v2,
    build_component_candidates_v2, run_copy_activation_patching_v2,
    summarize_activation_patching_v2, summarize_copy_specificity,
)


def _save_case_rows(adapter, output: Path, discovery, evaluation, control):
    rows = []
    for task, split, cases in (
        ("sequence_copy", "discovery", discovery),
        ("sequence_copy", "evaluation", evaluation),
        ("random_transition_control", "control", control),
    ):
        for case in cases:
            rows.append({
                "task": task, "split": split, "prompt_id": case.prompt_id,
                "input_token_ids_json": json.dumps(list(case.input_ids)),
                "decoded_prompt": adapter.decode(case.input_ids),
                "first_A_position": case.first_occurrence_position,
                "successor_B_position": case.successor_position,
                "query_position": case.query_position,
                "expected_token_id": case.expected_token_id,
                "decoded_expected_token": adapter.decode([case.expected_token_id]),
                "corruption_token_id": case.corruption_token_id,
                "decoded_corruption_token": adapter.decode([case.corruption_token_id]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "sequence_copy_cases_v2.csv", index=False)
    digest_input = "\n".join(
        f"{r['task']}\0{r['split']}\0{r['input_token_ids_json']}\0{r['expected_token_id']}"
        for r in rows
    ).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--discovery-cases", type=int, default=96)
    parser.add_argument("--evaluation-cases", type=int, default=128)
    parser.add_argument("--control-cases", type=int, default=128)
    parser.add_argument("--patch-cases", type=int, default=12)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--top-per-layer", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "sequence_copy_v2")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    counts = [args.discovery_cases, args.evaluation_cases, args.control_cases,
              args.patch_cases, args.layers, args.top_per_layer, args.batch_size]
    if min(counts) < 1:
        parser.error("All case, layer, batch and top-per-layer counts must be positive")

    seed_everything(args.seed)
    print(f"Loading {args.model_id}...")
    adapter = load_huggingface_model(args.model_id, args.device, args.dtype,
                                     args.local_files_only, args.revision)
    layers = choose_layer_indices(adapter.n_layers, args.layers)
    print(f"Loaded {adapter.architecture}: {adapter.n_layers} layers, {adapter.n_heads} query heads, "
          f"{adapter.n_kv_heads} KV heads, MLP width {adapter.intermediate_size}; selected layers {layers}")

    print("Building three disjoint token pools: discovery, copy evaluation, and random-transition control...")
    all_tokens = lexical_token_ids(adapter.tokenizer)
    discovery_pool, evaluation_pool, control_pool = split_three_disjoint_token_pools(all_tokens, args.seed)
    discovery = make_sequence_copy_cases(discovery_pool, args.discovery_cases, args.seed + 1)
    evaluation = make_sequence_copy_cases(evaluation_pool, args.evaluation_cases, args.seed + 2,
                                          start_prompt_id=100000)
    control = make_random_transition_control_cases(control_pool, args.control_cases, args.seed + 8)
    patch_cases = evaluation[:min(args.patch_cases, len(evaluation))]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = args.model_id.replace("/", "_").replace(":", "_")
    output = args.output_dir / f"{slug}_{stamp}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    case_digest = _save_case_rows(adapter, output, discovery, evaluation, control)

    print("1/7: Measuring final-query attention to the earlier successor B (A retained as diagnostic)...")
    attention_df = collect_copy_attention_scores_v2(adapter, discovery, layers, args.batch_size)
    attention_df.to_csv(output / "discovery_attention_scores_v2.csv", index=False)
    attention_summary = (attention_df.groupby(["layer", "head"], as_index=False)
        .agg(mean_attention_to_successor_B=("attention_to_successor_B", "mean"),
             mean_lift_to_successor_B=("attention_lift_to_successor_B", "mean"),
             std_lift_to_successor_B=("attention_lift_to_successor_B", "std"),
             mean_attention_to_first_A=("attention_to_first_A", "mean"),
             mean_lift_to_first_A=("attention_lift_to_first_A", "mean")))
    attention_summary.to_csv(output / "discovery_attention_summary_v2.csv", index=False)

    print("2/7: Ranking MLP neurons by clean-vs-corrupted final-query activation sensitivity...")
    neuron_scores = rank_mlp_neurons_v2(adapter, discovery, layers, args.batch_size)
    neuron_frame = pd.DataFrame([
        {"layer": layer, "neuron": neuron, "mean_abs_clean_corrupt_delta": delta,
         "mean_abs_activity": activity}
        for layer, ranked in neuron_scores.items()
        for neuron, delta, activity in ranked
    ])
    neuron_frame.to_csv(output / "discovery_neuron_ranking_v2.csv", index=False)

    print("3/7: Selecting B-attending heads and task-sensitive neurons with matched controls...")
    candidates = build_component_candidates_v2(
        attention_df, neuron_scores, layers, adapter.n_heads,
        adapter.intermediate_size, args.top_per_layer, args.seed + 3,
    )
    candidates.to_csv(output / "component_candidates_v2.csv", index=False)

    print("4/7: Evaluating candidate/control ablations on held-out sequence-copy prompts...")
    baseline, details = evaluate_copy_interventions(adapter, evaluation, candidates, args.batch_size)
    baseline.to_csv(output / "heldout_baseline_metrics.csv", index=False)
    baseline_summary = summarize_baseline_metrics(baseline, args.seed + 5)
    baseline_summary.to_csv(output / "heldout_baseline_summary.csv", index=False)
    details.to_csv(output / "heldout_ablation_per_prompt.csv", index=False)
    copy_summary, copy_paired = summarize_interventions(details, args.seed + 4)
    copy_summary.to_csv(output / "heldout_ablation_summary.csv", index=False)
    copy_paired.to_csv(output / "candidate_vs_random_controls.csv", index=False)

    print("5/7: Evaluating the same components on a repetition-free random-transition control task...")
    control_baseline, control_details = evaluate_copy_interventions(
        adapter, control, candidates, args.batch_size
    )
    control_baseline.to_csv(output / "control_baseline_metrics.csv", index=False)
    control_details.to_csv(output / "control_ablation_per_prompt.csv", index=False)
    control_summary, control_paired = summarize_interventions(control_details, args.seed + 9)
    control_summary.to_csv(output / "control_ablation_summary.csv", index=False)
    control_paired.to_csv(output / "control_candidate_vs_random.csv", index=False)
    specificity = summarize_copy_specificity(details, control_details, args.seed + 10)
    specificity.to_csv(output / "copy_specificity_summary.csv", index=False)

    print("6/7: Running non-trivial block-position and component-specific activation patching...")
    patching = run_copy_activation_patching_v2(adapter, patch_cases, layers, candidates, args.seed + 11)
    patching.to_csv(output / "activation_patching_per_prompt_v2.csv", index=False)
    patch_summary = summarize_activation_patching_v2(patching, args.seed + 12)
    patch_summary.to_csv(output / "activation_patching_summary_v2.csv", index=False)

    print("7/7: Writing provenance and protocol report...")
    target_acc = float(baseline.baseline_correct.mean()) if not baseline.empty else float("nan")
    control_acc = float(control_baseline.baseline_correct.mean()) if not control_baseline.empty else float("nan")
    metadata = {
        "protocol_version": "2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": getattr(adapter.config, "_commit_hash", None),
        "architecture": adapter.architecture,
        "n_layers": adapter.n_layers,
        "n_query_heads": adapter.n_heads,
        "n_kv_heads": adapter.n_kv_heads,
        "intermediate_size": adapter.intermediate_size,
        "device": str(adapter.device),
        "dtype": str(next(adapter.model.parameters()).dtype),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": args.seed,
        "selected_layers": layers,
        "discovery_cases": len(discovery),
        "evaluation_cases": len(evaluation),
        "control_cases": len(control),
        "patch_cases": len(patch_cases),
        "top_per_layer": args.top_per_layer,
        "batch_size": args.batch_size,
        "sequence_copy_and_control_cases_sha256": case_digest,
        "token_pools_pairwise_disjoint": (
            len(set(discovery_pool) & set(evaluation_pool)) == 0
            and len(set(discovery_pool) & set(control_pool)) == 0
            and len(set(evaluation_pool) & set(control_pool)) == 0
        ),
        "baseline_copy_accuracy": target_acc,
        "control_random_target_accuracy": control_acc,
        "method_notes": [
            "Primary induction-style head score is final-query attention lift to the earlier successor B position.",
            "Attention to the earlier A position is retained as a diagnostic, not the primary induction score.",
            "MLP neurons are ranked by final-query activation change under B-token corruption.",
            "Neuron random controls are sampled from coordinates with similar mean absolute activation magnitude.",
            "Head/neuron ablations are evaluated at the final query on held-out copy and random-transition control tasks.",
            "Activation patching includes the earlier successor position, a random-position block control, and component-specific head/neuron patches.",
            "No whole-block final-query patch is performed because it is trivially close to restoring the clean final state.",
            "Discovery, copy evaluation, and control tasks use pairwise-disjoint token identity pools.",
            "Bootstrap intervals and paired sign-flip tests are prompt-level; BH correction is applied across component-control pairs.",
        ],
        "limitations": [
            "The random-transition control is a generic next-token sensitivity control, not a natural-language benchmark.",
            "Attention weights are descriptive; causal interpretation requires intervention and patching evidence.",
            "Ablation can be off-manifold; interpret with matched controls and activation patching.",
            "Cross-model layer/head indices should not be equated directly across architectures.",
        ],
    }
    write_json(output / "run_metadata.json", metadata)
    top = copy_paired.sort_values("selected_minus_random_mean_margin_drop", ascending=False).head(12) if not copy_paired.empty else copy_paired
    spec_top = specificity.sort_values("copy_minus_control_specificity", ascending=False).head(12) if not specificity.empty else specificity
    report = [
        "# Sequence-copying mechanistic interpretability study — Protocol V2", "",
        f"- Model: `{args.model_id}`", f"- Architecture: `{adapter.architecture}`",
        f"- Selected layers: `{layers}`", f"- Discovery/copy-eval/control cases: `{len(discovery)}/{len(evaluation)}/{len(control)}`",
        f"- Patch cases: `{len(patch_cases)}`", f"- Held-out copy accuracy: **{target_acc:.4f}**",
        f"- Random-transition control top-1 accuracy: **{control_acc:.4f}**", "",
        "## Held-out baseline uncertainty", "", baseline_summary.to_markdown(index=False), "",
        "## Candidate vs random-control ablation effects on copy task", "",
        top.to_markdown(index=False) if not top.empty else "No paired results.", "",
        "## Copy-task specificity vs random-transition control", "",
        spec_top.to_markdown(index=False) if not spec_top.empty else "No specificity results.", "",
        "## Activation patching", "", patch_summary.to_markdown(index=False), "",
        "## Interpretation guidance", "",
        "A component is more specifically implicated in sequence copying when its selected-vs-control ablation effect is larger on copy prompts than on independent random-transition prompts, with uncertainty intervals and corrected tests supporting the difference.",
        "The successor-B attention score is a candidate-discovery signal only; it is not proof of an induction circuit.",
        "Component-specific patching and non-trivial position controls provide additional causal evidence, but should be interpreted alongside baseline performance and held-out ablations.",
    ]
    (output / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nProtocol V2 study complete. Outputs saved to: {output}")
    print(f"Held-out copy accuracy: {target_acc:.4f}; random-transition control accuracy: {control_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
