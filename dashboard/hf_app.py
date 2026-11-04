"""Standalone Hugging Face model explorer and research-results dashboard."""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from interpretability_lab.hf_backend import load_huggingface_model

st.set_page_config(page_title="HF Mechanistic Interpretability Lab", page_icon="🧪", layout="wide")
st.title("Hugging Face · Mechanistic Interpretability Lab")
st.caption("Inspect pretrained decoder-only Transformers and run causal interventions. Model weights are downloaded from Hugging Face on first load and are not bundled with this repository.")

with st.sidebar:
    st.header("Model")
    model_id = st.text_input("Hugging Face model ID", "HuggingFaceTB/SmolLM3-3B-Base")
    revision = st.text_input("Revision (optional branch / tag / commit)", "")
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    dtype = st.selectbox("Precision", ["auto", "bfloat16", "float16", "float32"], index=0)
    local_only = st.checkbox("Use cached files only", value=False)
    load_clicked = st.button("Load / switch model", type="primary", use_container_width=True)

if load_clicked:
    try:
        with st.spinner(f"Loading {model_id}. The first download can be several GB..."):
            st.session_state["hf_adapter"] = load_huggingface_model(
                model_id, device, dtype, local_only, revision or None
            )
            st.session_state["hf_loaded_id"] = model_id
            for stale_key in ("hf_top_df", "hf_inspect_ids", "hf_inspect_attentions", "hf_inspect_tokens", "hf_mlp_acts"):
                st.session_state.pop(stale_key, None)
        st.success(f"Loaded {model_id}")
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")

adapter = st.session_state.get("hf_adapter")
if adapter is None:
    st.info("Enter a Hugging Face model ID and click **Load / switch model**. The default is SmolLM3-3B-Base.")
    st.markdown("#### Hardware guidance")
    st.write("A 3B model requires several GB of memory for weights alone. FP32 CPU loading can require substantially more RAM than the compressed checkpoint size. Start with short prompts and a small layer sweep. A CUDA GPU is recommended; quantized loading can be added separately if your machine is memory-constrained.")
    st.stop()

cfg = adapter.config
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Layers", adapter.n_layers)
m2.metric("Query heads", adapter.n_heads)
m3.metric("KV heads", adapter.n_kv_heads)
m4.metric("Hidden size", adapter.hidden_size)
m5.metric("MLP width", adapter.intermediate_size)
st.caption(f"{adapter.model_id} · {adapter.architecture} · {adapter.device} · {next(adapter.model.parameters()).dtype}")

prompt = st.text_area("Prompt", "The capital of France is", height=90)
inspect = st.button("Inspect prompt", type="primary")
if inspect:
    try:
        ids = adapter.encode(prompt)
        with torch.inference_mode():
            output = adapter.forward(ids, output_attentions=True)
            probs = output.logits[0, -1].float().softmax(-1)
            vals, indices = torch.topk(probs, k=min(12, probs.numel()))
            top_df = pd.DataFrame({
                "token": [adapter.decode([int(i)]) for i in indices.cpu().tolist()],
                "token_id": indices.cpu().tolist(),
                "probability": vals.cpu().tolist(),
            })
            st.session_state["hf_inspect_ids"] = ids.detach().cpu()
            st.session_state["hf_inspect_attentions"] = [x.detach().float().cpu() for x in (output.attentions or [])]
            st.session_state["hf_inspect_tokens"] = [adapter.decode([int(t)]) for t in ids[0].tolist()]
            st.session_state["hf_top_df"] = top_df
    except Exception as exc:
        st.error(f"Inspection failed: {exc}")

if "hf_top_df" in st.session_state:
    st.subheader("Next-token distribution")
    st.dataframe(st.session_state["hf_top_df"], use_container_width=True, hide_index=True)
    attentions = st.session_state.get("hf_inspect_attentions", [])
    if attentions:
        st.subheader("Attention patterns")
        left, right = st.columns(2)
        layer = left.slider("Layer", 0, min(adapter.n_layers, len(attentions)) - 1, 0, key="hf_att_layer")
        head = right.slider("Query head", 0, adapter.n_heads - 1, 0, key="hf_att_head")
        matrix = attentions[layer][0, head].numpy()
        labels = st.session_state["hf_inspect_tokens"]
        fig = px.imshow(matrix, x=labels, y=labels, aspect="auto", labels={"x": "Key token", "y": "Query token", "color": "Attention"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Attention weights are descriptive. They do not establish a head's causal role by themselves.")
    else:
        st.warning("This model did not return attention tensors. Check eager-attention support in your Transformers version.")

    st.subheader("Gated MLP intermediate activations")
    mlp_layer = st.slider("MLP layer to inspect", 0, adapter.n_layers - 1, 0, key="hf_mlp_layer")
    if st.button("Capture this layer's MLP activations"):
        try:
            ids_for_mlp = st.session_state["hf_inspect_ids"].to(adapter.device)
            with st.spinner("Capturing the post-gating MLP intermediate vector..."):
                acts = adapter.capture_mlp_intermediates(ids_for_mlp, [int(mlp_layer)])[int(mlp_layer)]
            st.session_state["hf_mlp_acts"] = acts
            st.session_state["hf_mlp_layer_cached"] = int(mlp_layer)
        except Exception as exc:
            st.error(f"Could not capture MLP activations: {exc}")
    if "hf_mlp_acts" in st.session_state and st.session_state.get("hf_mlp_layer_cached") == int(mlp_layer):
        acts = st.session_state["hf_mlp_acts"][0].numpy()
        selected_neuron = st.slider("Neuron coordinate", 0, acts.shape[-1] - 1, int(np.argmax(np.abs(acts).mean(0))), key="hf_mlp_neuron")
        token_labels = st.session_state["hf_inspect_tokens"]
        st.plotly_chart(px.bar(x=token_labels, y=acts[:, selected_neuron], labels={"x": "Token position", "y": "Post-gating activation"}, title=f"Layer {mlp_layer}, intermediate coordinate {selected_neuron}"), use_container_width=True)
        top_neurons = np.argsort(np.abs(acts).mean(0))[-10:][::-1]
        st.dataframe(pd.DataFrame({"neuron": top_neurons, "mean_abs_activation": np.abs(acts).mean(0)[top_neurons]}), use_container_width=True)

st.divider()
st.header("Causal intervention lab")
st.write("Ablations modify the input to a projection: query-head context vectors before the attention output projection, or gated MLP coordinates before the down projection. This is designed to handle grouped-query attention correctly.")
if "hf_inspect_ids" not in st.session_state:
    st.info("Inspect a prompt above before running a single-prompt intervention.")
else:
    intervention_kind = st.selectbox("Component", ["Attention query head", "Gated MLP neuron"])
    col1, col2 = st.columns(2)
    intervention_layer = col1.number_input("Layer", min_value=0, max_value=adapter.n_layers - 1, value=0, step=1)
    max_component = adapter.n_heads - 1 if intervention_kind == "Attention query head" else adapter.intermediate_size - 1
    component = col2.number_input("Component index", min_value=0, max_value=max_component, value=0, step=1)
    if st.button("Run ablation"):
        kind = "head" if intervention_kind == "Attention query head" else "neuron"
        with st.spinner("Running baseline and intervened forward passes..."):
            try:
                row = adapter.ablation_metrics(st.session_state["hf_inspect_ids"].to(adapter.device), kind, int(intervention_layer), int(component))
                a, b, c = st.columns(3)
                a.metric("KL: baseline → ablated", f"{row['kl_baseline_to_intervened']:.6f}")
                b.metric("Top token changed", "Yes" if row["top_token_changed"] else "No")
                c.metric("Baseline-top logit Δ", f"{row['baseline_top_logit_change']:.4f}")
                st.json(row)
            except Exception as exc:
                st.error(f"Intervention failed: {exc}")

st.divider()
st.header("Activation patching")
patch_a, patch_b = st.columns(2)
clean_text = patch_a.text_input("Clean prompt", "The capital of France is", key="hf_clean_patch")
corrupt_text = patch_b.text_input("Corrupted prompt", "The capital of Germany is", key="hf_corrupt_patch")
if st.button("Run layer-wise activation patching"):
    try:
        clean_ids, corrupt_ids = adapter.encode(clean_text), adapter.encode(corrupt_text)
        with st.spinner("Capturing clean and corrupted residual states, then patching each selected layer..."):
            rows = adapter.activation_patch_sweep(clean_ids, corrupt_ids)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.plotly_chart(px.bar(df, x="layer", y="logit_recovery_fraction", title="Final-token target-logit recovery by patched layer"), use_container_width=True)
        st.download_button("Download patching CSV", df.to_csv(index=False), "hf_activation_patching.csv", "text/csv")
    except Exception as exc:
        st.error(f"Patching failed: {exc}")

st.divider()
st.header("Saved CLI research runs")
artifact_root = ROOT / "artifacts"
runs = sorted([p for p in artifact_root.glob("hf_*") if p.is_dir()], reverse=True)
if not runs:
    st.info("No Hugging Face CLI run found yet. Run `python scripts/run_hf_research.py` from the repository root.")
else:
    selected = st.selectbox("Select run", runs, format_func=lambda p: p.name)
    summary_path = selected / "ablation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, use_container_width=True)
        if not summary.empty:
            st.plotly_chart(px.bar(summary.head(25), x="component", y="kl_mean", color="kind", facet_row="layer", title="Largest measured KL changes"), use_container_width=True)
        st.download_button("Download ablation summary", summary_path.read_bytes(), summary_path.name, "text/csv")
    metadata_path = selected / "run_metadata.json"
    if metadata_path.exists():
        with st.expander("Run provenance and limitations"):
            st.json(json.loads(metadata_path.read_text(encoding="utf-8")))
    report_path = selected / "RUN_REPORT.md"
    if report_path.exists():
        with st.expander("Generated run report"):
            st.markdown(report_path.read_text(encoding="utf-8"))

comparison_path = artifact_root / "model_comparison" / "model_comparison.csv"
if comparison_path.exists():
    st.subheader("Cross-model comparison")
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True)
    st.download_button("Download cross-model comparison", comparison_path.read_bytes(), comparison_path.name, "text/csv")

st.divider()
st.header("Controlled sequence-copying research")
st.caption("Primary study: [prefix, A, B, fillers, A] → predict B. Attention is used for discovery; held-out ablations, random controls and activation patching provide causal evidence.")
sequence_root = artifact_root / "sequence_copy"
sequence_runs = sorted([p for p in sequence_root.glob("*") if p.is_dir()], reverse=True) if sequence_root.exists() else []
if not sequence_runs:
    st.info("No controlled sequence-copying study has been run yet. Start with `python scripts/run_sequence_copy_study.py --model-id gpt2`.")
else:
    selected_sequence_run = st.selectbox("Select sequence-copying run", sequence_runs, format_func=lambda p: p.name, key="sequence_copy_run")
    run_metadata_path = selected_sequence_run / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8")) if run_metadata_path.exists() else {}
    base_accuracy = run_metadata.get("baseline_copy_accuracy")
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Model", str(run_metadata.get("model_id", "unknown")))
    sm2.metric("Held-out prompts", str(run_metadata.get("evaluation_cases", "—")))
    sm3.metric("Selected layers", str(len(run_metadata.get("selected_layers", []))))
    sm4.metric("Baseline copy accuracy", f"{base_accuracy:.3f}" if isinstance(base_accuracy, (int, float)) else "—")

    paired_path = selected_sequence_run / "candidate_vs_random_controls.csv"
    if paired_path.exists():
        paired_results = pd.read_csv(paired_path)
        st.subheader("Discovery-ranked components versus random controls")
        if not paired_results.empty:
            st.dataframe(paired_results.sort_values("selected_minus_random_mean_margin_drop", ascending=False), use_container_width=True, hide_index=True)
            st.plotly_chart(px.scatter(
                paired_results, x="layer", y="selected_minus_random_mean_margin_drop",
                color="kind", error_y=paired_results.ci95_high - paired_results.selected_minus_random_mean_margin_drop,
                error_y_minus=paired_results.selected_minus_random_mean_margin_drop - paired_results.ci95_low,
                hover_data=["pair_id", "n_prompts", "permutation_p", "q_bh"],
                title="Selected-component excess copy-margin damage over same-layer random controls"
            ), use_container_width=True)
        else:
            st.warning("No paired candidate/control comparisons were produced.")
        st.download_button("Download paired-control statistics", paired_path.read_bytes(), paired_path.name, "text/csv")

    summary_path = selected_sequence_run / "heldout_ablation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.subheader("Held-out component interventions")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if not summary.empty:
            st.plotly_chart(px.bar(
                summary, x="component", y="margin_drop_mean", color="selection",
                facet_row="kind", facet_col="layer", title="Held-out copy-margin drop after final-query ablation"
            ), use_container_width=True)
        st.download_button("Download ablation summary", summary_path.read_bytes(), summary_path.name, "text/csv")

    patch_summary_path = selected_sequence_run / "activation_patching_summary.csv"
    if patch_summary_path.exists():
        patch_summary = pd.read_csv(patch_summary_path)
        st.subheader("Clean-to-corrupted activation patching")
        st.dataframe(patch_summary, use_container_width=True, hide_index=True)
        if not patch_summary.empty:
            st.plotly_chart(px.line(
                patch_summary, x="layer", y="mean_margin_recovery", color="patch_location",
                markers=True, title="Copy-margin recovery by patched layer and token position"
            ), use_container_width=True)
        st.download_button("Download activation-patching summary", patch_summary_path.read_bytes(), patch_summary_path.name, "text/csv")

    report_path = selected_sequence_run / "RUN_REPORT.md"
    if report_path.exists():
        with st.expander("Generated research report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    if run_metadata_path.exists():
        with st.expander("Run provenance and interpretation limitations"):
            st.json(run_metadata)
