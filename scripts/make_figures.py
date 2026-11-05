"""Regenerate the README results figure from the saved Protocol V2 SmolLM3 bundles.

Usage:  python scripts/make_figures.py
Requires: pandas, matplotlib
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "protocol_v2" / "smollm3"
SEEDS = (7, 13, 29)
COLORS = {7: "#1f77b4", 13: "#ff7f0e", 29: "#2ca02c"}


def load(seed: int, name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / f"seed_{seed}" / name)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Panel A: selected-head ablation effect vs matched random head, per screened layer.
    for i, seed in enumerate(SEEDS):
        df = load(seed, "candidate_vs_random_controls.csv")
        df = df[df.kind == "head"].sort_values("layer")
        x = df.layer + (i - 1) * 0.8
        err = [df.selected_minus_random_mean_margin_drop - df.ci95_low,
               df.ci95_high - df.selected_minus_random_mean_margin_drop]
        ax1.errorbar(x, df.selected_minus_random_mean_margin_drop, yerr=err, fmt="o",
                     ms=4, capsize=2, color=COLORS[seed], label=f"seed {seed}")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.annotate("L15 H11", xy=(15, 0.31), xytext=(18, 0.36), arrowprops=dict(arrowstyle="-", color="k"))
    ax1.set_xlabel("Layer (8 of 36 screened)")
    ax1.set_ylabel("Extra copy-margin drop vs. random head")
    ax1.set_title("Ablating the discovery-selected head (95% CI)")
    ax1.legend(frameon=False)

    # Panel B: block-state patching at the earlier successor position vs random-position control.
    for seed in SEEDS:
        df = load(seed, "activation_patching_summary_v2.csv")
        for loc, ls in (("block_successor_position", "-"), ("block_random_position", ":")):
            d = df[df.patch_location == loc].sort_values("layer")
            ax2.plot(d.layer, d.mean_margin_recovery, ls, color=COLORS[seed], marker="o", ms=3,
                     label=f"seed {seed}" if loc.endswith("successor_position") else None)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Fraction of copy-margin gap recovered")
    ax2.set_title("Clean block-state patch (solid) vs random position (dotted)")
    ax2.legend(frameon=False)

    out = ROOT / "docs" / "figures" / "smollm3_v2_summary.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
