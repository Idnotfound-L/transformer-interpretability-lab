#!/usr/bin/env python3
"""Compare two completed Hugging Face experiment runs on the same evaluation prompts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def load_run(path: Path):
    meta_path = path / "run_metadata.json"
    summary_path = path / "ablation_summary.csv"
    if not meta_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"{path} must contain run_metadata.json and ablation_summary.csv")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    return meta, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_comparison"))
    args = parser.parse_args()
    meta_a, a = load_run(args.run_a)
    meta_b, b = load_run(args.run_b)
    hash_a = meta_a.get("evaluation_prompt_sha256")
    hash_b = meta_b.get("evaluation_prompt_sha256")
    if hash_a != hash_b:
        raise SystemExit(
            "Evaluation prompt hashes differ. Re-run both models with the exact same "
            "--evaluation-prompts file before comparing them."
        )

    rows = []
    for label, frame in ((meta_a.get("model", args.run_a.name), a), (meta_b.get("model", args.run_b.name), b)):
        for kind, group in frame.groupby("kind"):
            rows.append({
                "model": label,
                "kind": kind,
                "components_evaluated": int(len(group)),
                "mean_component_kl": float(group["kl_mean"].mean()),
                "median_component_kl": float(group["kl_mean"].median()),
                "top_component_kl": float(group["kl_mean"].max()),
                "mean_top_token_change_rate": float(group["top_token_change_rate"].mean()),
            })
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "model_comparison.csv", index=False)
    (args.output / "comparison_metadata.json").write_text(json.dumps({
        "model_a": meta_a.get("model"),
        "model_b": meta_b.get("model"),
        "evaluation_prompt_sha256": hash_a,
        "interpretation": "Compare aggregate intervention sensitivity only when both runs use identical evaluation prompts. Layer/head identities are not assumed to align across architectures.",
    }, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Saved comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
