"""Offline dashboard for Protocol V2 sequence-copy experiments."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts" / "sequence_copy_v2"
st.set_page_config(page_title="Protocol V2 · Mechanistic Interpretability", page_icon="🧪", layout="wide")
st.title("Mechanistic Interpretability Lab — Protocol V2")
st.caption("Offline viewer for saved Protocol V2 runs. No model weights are loaded by this dashboard.")
run_root_text = st.sidebar.text_input("Run directory", str(DEFAULT_ROOT))
run_root = Path(run_root_text).expanduser()
runs = []
if run_root.exists():
    for meta_path in run_root.rglob("run_metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("protocol_version")) == "2.0":
                runs.append((meta_path.parent, meta))
        except Exception:
            pass
runs.sort(key=lambda item: (str(item[1].get("model_id")), int(item[1].get("seed", -1))))
if not runs:
    st.info("No Protocol V2 run folders found. Run scripts/run_sequence_copy_study_v2.py first, then point this dashboard at the output directory.")
    st.stop()
labels = [f"{m.get('model_id')} · seed {m.get('seed')} · {p.name}" for p, m in runs]
idx = st.sidebar.selectbox("Experiment", range(len(runs)), format_func=lambda i: labels[i])
folder, meta = runs[idx]

c = st.columns(6)
c[0].metric("Model", str(meta.get("model_id", "—")))
c[1].metric("Seed", str(meta.get("seed", "—")))
c[2].metric("Device", str(meta.get("device", "—")))
c[3].metric("Discovery", str(meta.get("discovery_cases", "—")))
c[4].metric("Copy eval", str(meta.get("evaluation_cases", "—")))
c[5].metric("Control", str(meta.get("control_cases", "—")))
st.caption(f"Protocol {meta.get('protocol_version')} · dtype {meta.get('dtype')} · revision {meta.get('resolved_revision') or meta.get('requested_revision') or 'not pinned'}")
with st.expander("Run provenance and limitations"):
    st.json(meta)

def load(name: str):
    path = folder / name
    return pd.read_csv(path) if path.exists() else None

baseline = load("heldout_baseline_summary.csv")
if baseline is not None and not baseline.empty:
    st.subheader("Held-out baseline")
    st.dataframe(baseline, use_container_width=True, hide_index=True)

att = load("discovery_attention_summary_v2.csv")
if att is not None and not att.empty:
    st.subheader("Attention discovery — successor B")
    score = "mean_lift_to_successor_B" if "mean_lift_to_successor_B" in att.columns else "mean_attention_to_successor_B"
    view = att.sort_values(score, ascending=False)
    st.dataframe(view.head(40), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(view.head(40), x="head", y=score, color="layer", title="Discovery-ranked attention heads"), use_container_width=True)

neurons = load("discovery_neuron_ranking_v2.csv")
if neurons is not None and not neurons.empty:
    st.subheader("MLP neuron discovery")
    layer_filter = st.multiselect("MLP layers", sorted(neurons.layer.unique()), default=sorted(neurons.layer.unique()))
    visible = neurons[neurons.layer.isin(layer_filter)]
    st.dataframe(visible.head(100), use_container_width=True, hide_index=True)

cand = load("component_candidates_v2.csv")
if cand is not None and not cand.empty:
    st.subheader("Selected candidates and matched controls")
    st.dataframe(cand, use_container_width=True, hide_index=True)

paired = load("candidate_vs_random_controls.csv")
if paired is not None and not paired.empty:
    st.subheader("Held-out copy-task ablation: candidate vs matched random control")
    sort_col = "q_bh" if "q_bh" in paired.columns else paired.columns[0]
    st.dataframe(paired.sort_values(sort_col), use_container_width=True, hide_index=True)
    if "selected_minus_random_mean_margin_drop" in paired.columns:
        st.plotly_chart(px.scatter(paired, x="layer", y="selected_minus_random_mean_margin_drop", color="kind",
                                   hover_data=[x for x in ["pair_id", "n_prompts", "permutation_p", "q_bh"] if x in paired.columns],
                                   title="Additional margin drop from selected component"), use_container_width=True)

specificity = load("copy_specificity_summary.csv")
if specificity is not None and not specificity.empty:
    st.subheader("Copy-task specificity vs random-transition control")
    st.dataframe(specificity.sort_values("q_bh") if "q_bh" in specificity.columns else specificity,
                 use_container_width=True, hide_index=True)
    if "copy_minus_control_specificity" in specificity.columns:
        st.plotly_chart(px.scatter(specificity, x="layer", y="copy_minus_control_specificity", color="kind",
                                   title="Copy effect minus random-transition-control effect"), use_container_width=True)

patch = load("activation_patching_summary_v2.csv")
if patch is not None and not patch.empty:
    st.subheader("Activation patching")
    st.dataframe(patch, use_container_width=True, hide_index=True)
    if {"layer", "mean_margin_recovery", "patch_location"}.issubset(patch.columns):
        st.plotly_chart(px.line(patch, x="layer", y="mean_margin_recovery", color="patch_location", markers=True,
                                title="Mean margin recovery by patch location"), use_container_width=True)

with st.expander("Generated run report"):
    report = folder / "RUN_REPORT.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.warning("RUN_REPORT.md is not present in this run folder.")

st.divider()
st.caption("Attention is a discovery signal, not proof of function. Interpret ablations with baseline capability, matched controls, copy-task specificity, patching, and off-manifold limitations.")
