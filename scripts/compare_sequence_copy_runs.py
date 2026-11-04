#!/usr/bin/env python3
"""Create a descriptive cross-model summary from sequence-copy study run folders.

Usage:
  python scripts/compare_sequence_copy_runs.py artifacts/sequence_copy/gpt2_RUN artifacts/sequence_copy/SmolLM3_RUN
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="One or more sequence-copy run folders")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sequence_copy/model_comparison"))
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Supply at least two run folders to compare")

    records = []
    metadata_list = []
    for run in args.runs:
        metadata_path = run / "run_metadata.json"
        paired_path = run / "candidate_vs_random_controls.csv"
        patch_path = run / "activation_patching_summary.csv"
        if not metadata_path.exists() or not paired_path.exists():
            raise FileNotFoundError(f"Run folder is missing metadata or paired-control results: {run}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paired = pd.read_csv(paired_path)
        patch = pd.read_csv(patch_path) if patch_path.exists() else pd.DataFrame()
        metadata_list.append(metadata)
        for kind in ("head", "neuron"):
            subset = paired[paired.kind == kind] if not paired.empty else pd.DataFrame()
            significant = int((subset.q_bh < 0.05).sum()) if not subset.empty and "q_bh" in subset else 0
            records.append({
                "model_id": metadata.get("model_id", run.name),
                "kind": kind,
                "baseline_copy_accuracy": metadata.get("baseline_copy_accuracy"),
                "n_component_pairs": int(len(subset)),
                "mean_candidate_minus_random_margin_drop": float(subset.selected_minus_random_mean_margin_drop.mean()) if not subset.empty else np.nan,
                "component_pairs_q_bh_below_0_05": significant,
                "mean_paired_q_value": float(subset.q_bh.mean()) if not subset.empty and "q_bh" in subset else np.nan,
                "mean_successor_patch_margin_recovery": float(patch.loc[patch.patch_location == "corrupted_successor", "mean_margin_recovery"].mean()) if not patch.empty else np.nan,
                "mean_query_patch_margin_recovery": float(patch.loc[patch.patch_location == "final_query", "mean_margin_recovery"].mean()) if not patch.empty else np.nan,
                "run_folder": str(run),
            })

    # Design comparability is checked explicitly. Model-specific token IDs are not aligned.
    design_fields = ["seed", "discovery_cases", "evaluation_cases", "patch_cases", "top_per_layer", "selected_layer_count", "batch_size"]
    reference = metadata_list[0]
    design_match = all(all(meta.get(field) == reference.get(field) for field in design_fields)
                       for meta in metadata_list[1:])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame["shared_design_settings"] = design_match
    frame.to_csv(output / "sequence_copy_cross_model_summary.csv", index=False)

    notes = [
        "# Cross-model sequence-copying summary", "",
        f"- Compared runs: {len(args.runs)}",
        f"- Shared design settings across runs: {'yes' if design_match else 'no'}",
        "- Token IDs are tokenizer-specific and are not treated as aligned across models.",
        "- The comparison is descriptive; component indices and layer numbers do not imply functional correspondence.",
        "- Candidate-vs-control statistics are model-local. Do not compare q-values across models as if they were one test family.",
        "", "## Summary", "", frame.to_markdown(index=False), "",
    ]
    (output / "CROSS_MODEL_REPORT.md").write_text("\n".join(notes), encoding="utf-8")
    print(f"Saved cross-model summary to {output}")
    if not design_match:
        print("WARNING: study design settings differ across runs; interpret comparisons cautiously.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
