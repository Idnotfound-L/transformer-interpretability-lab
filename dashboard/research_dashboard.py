"""Offline dashboard for saved controlled sequence-copying research runs.

This app does not load model weights. Run the CLI research study first, then
launch this dashboard to inspect the saved measurements and provenance.
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "artifacts" / "sequence_copy"

st.set_page_config(page_title="Mechanistic Interpretability Research", page_icon="🧪", layout="wide")
st.title("Mechanistic Interpretability · Research Results")
st.caption("Offline viewer for reproducible sequence-copying studies. It reads saved CSV/JSON artifacts and does not download model weights.")

runs = sorted([path for path in RUNS_ROOT.glob("*") if path.is_dir() and (path / "run_metadata.json").exists()], reverse=True) if RUNS_ROOT.exists() else []
if not runs:
    st.info("No saved sequence-copying studies found. Run `python scripts/run_sequence_copy_study.py --model-id gpt2` first.")
    st.stop()

selected = st.sidebar.selectbox("Study run", runs, format_func=lambda path: path.name)
metadata = json.loads((selected / "run_metadata.json").read_text(encoding="utf-8"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model", str(metadata.get("model_id", "unknown")))
c2.metric("Layers sampled", len(metadata.get("selected_layers", [])))
c3.metric("Discovery prompts", metadata.get("discovery_cases", "—"))
c4.metric("Held-out prompts", metadata.get("evaluation_cases", "—"))
accuracy = metadata.get("baseline_copy_accuracy")
c5.metric("Baseline copy accuracy", f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "—")
st.caption(f"Run: {selected.name} · revision: {metadata.get('resolved_revision') or metadata.get('requested_revision') or 'not pinned'} · device: {metadata.get('device')} · dtype: {metadata.get('dtype')}")

with st.expander("Study design, provenance and limitations"):
    st.json(metadata)

baseline_summary_path = selected / "heldout_baseline_summary.csv"
if baseline_summary_path.exists():
    st.subheader("Held-out baseline performance and uncertainty")
    st.dataframe(pd.read_csv(baseline_summary_path), use_container_width=True, hide_index=True)

attention_path = selected / "discovery_attention_summary.csv"
if attention_path.exists():
    st.subheader("Discovery: attention to the earlier repeated token")
    attention = pd.read_csv(attention_path)
    if not attention.empty:
        attention = attention.sort_values("mean_attention_lift", ascending=False)
        st.dataframe(attention.head(30), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(
            attention.head(30), x="head", y="mean_attention_lift", color="layer",
            title="Top discovery-ranked attention heads by mean attention lift"
        ), use_container_width=True)
    st.download_button("Download all discovery attention scores", (selected / "discovery_attention_scores.csv").read_bytes(), "discovery_attention_scores.csv", "text/csv")

candidate_path = selected / "component_candidates.csv"
if candidate_path.exists():
    st.subheader("Selected components and matched controls")
    candidates = pd.read_csv(candidate_path)
    st.dataframe(candidates, use_container_width=True, hide_index=True)

summary_path = selected / "heldout_ablation_summary.csv"
paired_path = selected / "candidate_vs_random_controls.csv"
if summary_path.exists():
    st.subheader("Held-out causal interventions")
    summary = pd.read_csv(summary_path)
    if not summary.empty:
        kind_filter = st.multiselect("Component type", sorted(summary.kind.unique()), default=sorted(summary.kind.unique()))
        selection_filter = st.multiselect("Selection", sorted(summary.selection.unique()), default=sorted(summary.selection.unique()))
        visible = summary[summary.kind.isin(kind_filter) & summary.selection.isin(selection_filter)]
        st.dataframe(visible, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(
            visible, x="component", y="margin_drop_mean", color="selection",
            error_y=visible.margin_drop_ci95_high - visible.margin_drop_mean,
            error_y_minus=visible.margin_drop_mean - visible.margin_drop_ci95_low,
            facet_row="kind", facet_col="layer",
            title="Held-out target-margin drop after final-query ablation"
        ), use_container_width=True)
    st.download_button("Download per-component ablation summary", summary_path.read_bytes(), summary_path.name, "text/csv")

if paired_path.exists():
    st.subheader("Discovery-ranked candidate vs same-layer random control")
    paired = pd.read_csv(paired_path)
    if paired.empty:
        st.info("No complete candidate/control pairs were produced.")
    else:
        st.dataframe(paired.sort_values("q_bh"), use_container_width=True, hide_index=True)
        st.plotly_chart(px.scatter(
            paired, x="layer", y="selected_minus_random_mean_margin_drop", color="kind",
            error_y=paired.ci95_high - paired.selected_minus_random_mean_margin_drop,
            error_y_minus=paired.selected_minus_random_mean_margin_drop - paired.ci95_low,
            hover_data=["pair_id", "n_prompts", "permutation_p", "q_bh"],
            title="Excess copy-margin damage from discovery-ranked candidates"
        ), use_container_width=True)
    st.download_button("Download paired-control statistics", paired_path.read_bytes(), paired_path.name, "text/csv")

patch_summary_path = selected / "activation_patching_summary.csv"
patch_detail_path = selected / "activation_patching_per_prompt.csv"
if patch_summary_path.exists():
    st.subheader("Activation patching")
    patch_summary = pd.read_csv(patch_summary_path)
    st.dataframe(patch_summary, use_container_width=True, hide_index=True)
    if not patch_summary.empty:
        st.plotly_chart(px.line(
            patch_summary, x="layer", y="mean_margin_recovery", color="patch_location",
            markers=True, error_y=patch_summary.margin_recovery_ci95_high - patch_summary.mean_margin_recovery,
            error_y_minus=patch_summary.mean_margin_recovery - patch_summary.margin_recovery_ci95_low,
            title="Copy-margin recovery after clean activation patching"
        ), use_container_width=True)
    st.download_button("Download patching summary", patch_summary_path.read_bytes(), patch_summary_path.name, "text/csv")
    if patch_detail_path.exists():
        st.download_button("Download per-prompt patching data", patch_detail_path.read_bytes(), patch_detail_path.name, "text/csv")

cases_path = selected / "sequence_copy_cases.csv"
if cases_path.exists():
    with st.expander("Inspect exact discovery and evaluation prompts"):
        cases = pd.read_csv(cases_path)
        st.dataframe(cases, use_container_width=True, hide_index=True)
        st.download_button("Download exact prompt dataset", cases_path.read_bytes(), cases_path.name, "text/csv")

report_path = selected / "RUN_REPORT.md"
if report_path.exists():
    st.subheader("Generated research report")
    st.markdown(report_path.read_text(encoding="utf-8"))

st.divider()
st.caption("Interpretation rule: attention is a discovery signal. Stronger causal evidence comes from held-out interventions, same-layer random controls and activation patching. Report null results and limitations.")

comparison_path = RUNS_ROOT / "model_comparison" / "sequence_copy_cross_model_summary.csv"
if comparison_path.exists():
    st.divider()
    st.header("Cross-model summary")
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.download_button("Download cross-model summary", comparison_path.read_bytes(), comparison_path.name, "text/csv")
