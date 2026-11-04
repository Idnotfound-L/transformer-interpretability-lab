#!/usr/bin/env python3
"""Run the controlled sequence-copying mechanistic interpretability study."""
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
    lexical_token_ids, split_disjoint_token_pools, make_sequence_copy_cases,
    choose_layer_indices, collect_copy_attention_scores, rank_mlp_neurons,
    build_component_candidates, evaluate_copy_interventions,
    summarize_interventions, run_copy_activation_patching,
    summarize_baseline_metrics, summarize_activation_patching,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="gpt2", help="Start with gpt2; use HuggingFaceTB/SmolLM3-3B-Base for larger-model validation")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--discovery-cases", type=int, default=96)
    parser.add_argument("--evaluation-cases", type=int, default=128)
    parser.add_argument("--patch-cases", type=int, default=12)
    parser.add_argument("--layers", type=int, default=8, help="Evenly spaced layers across full model depth")
    parser.add_argument("--top-per-layer", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "sequence_copy")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if min(args.discovery_cases, args.evaluation_cases, args.patch_cases, args.layers, args.batch_size, args.top_per_layer) < 1:
        parser.error("All case, layer, batch and top-per-layer counts must be positive")

    seed_everything(args.seed)
    print(f"Loading {args.model_id}...")
    adapter = load_huggingface_model(args.model_id, args.device, args.dtype,
                                     args.local_files_only, args.revision)
    layers = choose_layer_indices(adapter.n_layers, args.layers)
    print(f"Loaded {adapter.architecture}: {adapter.n_layers} layers, {adapter.n_heads} query heads, "
          f"{adapter.n_kv_heads} KV heads, MLP width {adapter.intermediate_size}; selected layers {layers}")

    print("Building token-disjoint discovery/evaluation sequence-copy prompts...")
    all_tokens = lexical_token_ids(adapter.tokenizer)
    discovery_pool, evaluation_pool = split_disjoint_token_pools(all_tokens, args.seed)
    discovery = make_sequence_copy_cases(discovery_pool, args.discovery_cases, args.seed + 1)
    evaluation = make_sequence_copy_cases(evaluation_pool, args.evaluation_cases, args.seed + 2,
                                          start_prompt_id=100000)
    patch_cases = evaluation[:min(args.patch_cases, len(evaluation))]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = args.model_id.replace("/", "_").replace(":", "_")
    output = args.output_dir / f"{slug}_{stamp}"
    output.mkdir(parents=True, exist_ok=True)

    # Persist the exact token-ID prompts and readable decoded forms for auditability.
    case_rows = []
    for split_name, case_list in (("discovery", discovery), ("evaluation", evaluation)):
        for case in case_list:
            case_rows.append({
                "split": split_name,
                "prompt_id": case.prompt_id,
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
    cases_df = pd.DataFrame(case_rows)
    cases_df.to_csv(output / "sequence_copy_cases.csv", index=False)
    case_digest = hashlib.sha256(
        cases_df["split"].astype(str).str.cat(cases_df["input_token_ids_json"].astype(str), sep="\0").encode("utf-8")
    ).hexdigest()

    print("1/6: Measuring discovery attention patterns...")
    attention_df = collect_copy_attention_scores(adapter, discovery, layers, args.batch_size)
    attention_df.to_csv(output / "discovery_attention_scores.csv", index=False)
    attention_summary = (attention_df.groupby(["layer", "head"], as_index=False)
                         .agg(mean_attention_to_first_A=("attention_to_first_A", "mean"),
                              mean_attention_lift=("attention_lift", "mean"),
                              std_attention_lift=("attention_lift", "std")))
    attention_summary.to_csv(output / "discovery_attention_summary.csv", index=False)

    print("2/6: Ranking discovery-set MLP coordinates...")
    neuron_scores = rank_mlp_neurons(adapter, discovery, layers, args.batch_size)
    neuron_summary = pd.DataFrame([
        {"layer": layer, "neuron": neuron, "mean_abs_final_activation": score}
        for layer, scored in neuron_scores.items() for neuron, score in scored
    ])
    neuron_summary.to_csv(output / "discovery_neuron_ranking.csv", index=False)

    print("3/6: Selecting candidates and matched random controls...")
    candidates = build_component_candidates(
        attention_df, neuron_scores, layers, adapter.n_heads,
        adapter.intermediate_size, args.top_per_layer, args.seed + 3,
    )
    candidates.to_csv(output / "component_candidates.csv", index=False)

    print("4/6: Evaluating held-out causal ablations...")
    baseline, details = evaluate_copy_interventions(adapter, evaluation, candidates, args.batch_size)
    baseline.to_csv(output / "heldout_baseline_metrics.csv", index=False)
    baseline_summary = summarize_baseline_metrics(baseline, args.seed + 5)
    baseline_summary.to_csv(output / "heldout_baseline_summary.csv", index=False)
    details.to_csv(output / "heldout_ablation_per_prompt.csv", index=False)
    summary, paired = summarize_interventions(details, args.seed + 4)
    summary.to_csv(output / "heldout_ablation_summary.csv", index=False)
    paired.to_csv(output / "candidate_vs_random_controls.csv", index=False)

    print("5/6: Running clean-to-corrupted activation patching...")
    patch_df = run_copy_activation_patching(adapter, patch_cases, layers)
    patch_df.to_csv(output / "activation_patching_per_prompt.csv", index=False)
    patch_summary = summarize_activation_patching(patch_df, args.seed + 6)
    patch_summary.to_csv(output / "activation_patching_summary.csv", index=False)

    print("6/6: Writing provenance and measured-results report...")
    target_acc = float(baseline.baseline_correct.mean()) if not baseline.empty else float("nan")
    metadata = {
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
        "selected_layer_count": len(layers),
        "discovery_cases": len(discovery),
        "evaluation_cases": len(evaluation),
        "patch_cases": len(patch_cases),
        "top_per_layer": args.top_per_layer,
        "batch_size": args.batch_size,
        "sequence_copy_cases_sha256": case_digest,
        "token_pool_disjoint": len(set(discovery_pool).intersection(evaluation_pool)) == 0,
        "discovery_token_pool_size": len(discovery_pool),
        "evaluation_token_pool_size": len(evaluation_pool),
        "baseline_copy_accuracy": target_acc,
        "method_notes": [
            "Task is [random prefix, A, B, random fillers, A] -> predict B, fixed nine-token length.",
            "Discovery and evaluation prompts use disjoint token identity pools.",
            "Attention scores rank candidate heads descriptively; held-out ablations supply causal sensitivity evidence.",
            "Head and neuron interventions are restricted to the final query position.",
            "Each layer has discovery-ranked candidates paired with same-layer random component controls.",
            "Activation patching restores clean block outputs at the corrupted successor-token position or final query position.",
            "Bootstrap intervals and paired sign-flip tests are prompt-level; BH correction is applied across paired candidate-control tests.",
        ],
        "limitations": [
            "Synthetic sequence-copying behaviour is a narrow probe, not a complete account of language understanding.",
            "Ablation is potentially off-manifold; interpret alongside activation patching and controls.",
            "Layer screening may be subsampled when --layers is smaller than model depth.",
            "No claim of a learned induction circuit should be made unless baseline task performance and causal evidence support it.",
        ],
    }
    write_json(output / "run_metadata.json", metadata)

    top_lines = []
    if not paired.empty:
        top_lines = paired.sort_values("selected_minus_random_mean_margin_drop", ascending=False).head(12).to_markdown(index=False).splitlines()
    report = [
        "# Sequence-copying mechanistic interpretability study", "",
        f"- Model: `{args.model_id}`", f"- Architecture: `{adapter.architecture}`",
        f"- Layers screened: `{layers}`", f"- Discovery cases: `{len(discovery)}`",
        f"- Held-out evaluation cases: `{len(evaluation)}`", f"- Patch cases: `{len(patch_cases)}`",
        f"- Baseline held-out copy accuracy: **{target_acc:.4f}**", "",
        "## Held-out baseline uncertainty", "",
        baseline_summary.to_markdown(index=False), "",
        "## Paired candidate-vs-random-control results", "",
    ]
    report.extend(top_lines if top_lines else ["No paired candidate-control results were produced."])
    report += ["", "## Interpretation guidance", "",
               "A positive selected-minus-random margin-drop estimate means ablating the discovery-ranked component "
               "reduced the correct-token margin more than ablating its same-layer random control. Inspect the "
               "confidence interval and BH-adjusted q-value before interpreting the result. A negative result is "
               "still informative and should be reported honestly.", "",
               "Attention-to-first-A is a descriptive ranking signal, not causal proof. Causal evidence comes from "
               "held-out final-query ablations and clean-to-corrupted activation patching.", ""]
    (output / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(f"\nStudy complete. Outputs saved to: {output}")
    print(f"Held-out copy accuracy: {target_acc:.4f}")
    print("Inspect RUN_REPORT.md and candidate_vs_random_controls.csv before making research claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
