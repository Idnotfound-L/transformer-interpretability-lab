#!/usr/bin/env python3
"""Index Protocol V2 run folders and write a descriptive cross-seed summary.

This script does not combine p-values across seeds. It indexes per-run outputs and
computes descriptive averages for matched components only; the report explicitly
labels the result as descriptive rather than a pooled inferential test.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

REQUIRED = [
    "run_metadata.json", "RUN_REPORT.md", "heldout_baseline_summary.csv",
    "candidate_vs_random_controls.csv", "control_candidate_vs_random.csv",
    "copy_specificity_summary.csv", "activation_patching_summary_v2.csv",
]

def inspect_run(folder: Path) -> dict | None:
    meta_path = folder / "run_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(meta.get("protocol_version")) != "2.0":
        return None
    row = {
        "run_folder": str(folder),
        "model_id": meta.get("model_id"),
        "seed": meta.get("seed"),
        "device": meta.get("device"),
        "dtype": meta.get("dtype"),
        "resolved_revision": meta.get("resolved_revision"),
        "discovery_cases": meta.get("discovery_cases"),
        "evaluation_cases": meta.get("evaluation_cases"),
        "control_cases": meta.get("control_cases"),
        "patch_cases": meta.get("patch_cases"),
        "selected_layers": ",".join(map(str, meta.get("selected_layers", []))),
        "complete_required_files": all((folder / name).exists() for name in REQUIRED),
    }
    row["missing_files"] = ";".join(name for name in REQUIRED if not (folder / name).exists())
    return row

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Directory containing Protocol V2 run folders")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    out = (args.output_dir or root / "cross_seed_summary").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta in root.rglob("run_metadata.json"):
        row = inspect_run(meta.parent)
        if row:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No Protocol V2 runs found under {root}")
    index = pd.DataFrame(rows).sort_values(["model_id", "seed"])
    index.to_csv(out / "run_index.csv", index=False)

    baseline_rows = []
    paired_rows = []
    specificity_rows = []
    for row in rows:
        folder = Path(row["run_folder"])
        meta = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        seed = meta.get("seed")
        model = meta.get("model_id")
        bpath = folder / "heldout_baseline_summary.csv"
        if bpath.exists():
            df = pd.read_csv(bpath)
            for _, r in df.iterrows():
                baseline_rows.append({"model_id": model, "seed": seed, **r.to_dict()})
        ppath = folder / "candidate_vs_random_controls.csv"
        if ppath.exists():
            df = pd.read_csv(ppath)
            for _, r in df.iterrows():
                paired_rows.append({"model_id": model, "seed": seed, **r.to_dict()})
        spath = folder / "copy_specificity_summary.csv"
        if spath.exists():
            df = pd.read_csv(spath)
            for _, r in df.iterrows():
                specificity_rows.append({"model_id": model, "seed": seed, **r.to_dict()})
    if baseline_rows:
        pd.DataFrame(baseline_rows).to_csv(out / "baseline_by_seed.csv", index=False)
    if paired_rows:
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(out / "candidate_vs_random_by_seed.csv", index=False)
        if {"kind", "layer", "component", "selected_minus_random_mean_margin_drop"}.issubset(paired.columns):
            grouped = (paired.groupby(["model_id", "kind", "layer", "component"], dropna=False)
                       .agg(seed_count=("seed", "nunique"),
                            mean_effect=("selected_minus_random_mean_margin_drop", "mean"),
                            min_effect=("selected_minus_random_mean_margin_drop", "min"),
                            max_effect=("selected_minus_random_mean_margin_drop", "max"))
                       .reset_index())
            grouped.to_csv(out / "descriptive_component_consistency.csv", index=False)
    if specificity_rows:
        pd.DataFrame(specificity_rows).to_csv(out / "specificity_by_seed.csv", index=False)

    counts = index.groupby("model_id")["seed"].nunique().to_dict()
    lines = ["# Protocol V2 cross-seed run index", "",
             f"Indexed {len(index)} Protocol V2 run(s) under `{root}`.", "",
             "## Runs", "", index.to_markdown(index=False), "",
             "## Distinct seeds per model", ""]
    for model, count in counts.items():
        lines.append(f"- `{model}`: {count} distinct seed(s)")
    lines += ["", "## Interpretation", "",
              "The exported component-consistency table is descriptive only. It averages reported per-run estimates for the same model/kind/layer/component key; it does not pool prompt-level observations or combine p-values. Use the per-run confidence intervals and q-values in each source report. Ensure model revisions and prompt hashes are compatible before making cross-run claims.", ""]
    (out / "CROSS_SEED_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Indexed {len(index)} Protocol V2 run(s). Outputs: {out}")
    print(index[["model_id", "seed", "complete_required_files", "missing_files"]].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
