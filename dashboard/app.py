from pathlib import Path
import json
import sys
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from interpretability_lab.core import (
    GPTConfig,CharTokenizer,TinyGPT,DEMO_TEXT,collect_attention,
    collect_residuals,collect_mlp_hidden,pca_projection,
    attention_entropy,mean_attention_distance,run_ablation
)
from interpretability_lab.research import (
    induction_score_table, make_induction_prompt, activation_patch_sweep,
    evaluate_ablation_prompts, summarize_ablation_rows, sweep_components
)

st.set_page_config(page_title="Mechanistic Interpretability Lab", layout="wide")
st.title("Transformer Internals & Mechanistic Interpretability Lab")
st.caption("An experimental workspace for inspecting Transformer internals, testing causal hypotheses and reviewing reproducible outputs. TinyGPT is the local demonstration backend; GPT-2 Small is an optional pretrained backend.")

@st.cache_resource
def load_model():
    path=ROOT/"artifacts"/"tiny_gpt.pt"
    if not path.exists(): return None,None,None
    ck=torch.load(path,map_location="cpu",weights_only=False)
    tok=CharTokenizer(chars=ck.get("chars", sorted(set(DEMO_TEXT))))
    cfg=GPTConfig(**ck["config"])
    m=TinyGPT(cfg); m.load_state_dict(ck["model_state"])
    m.eval()
    return m,tok,ck

model,tokenizer,checkpoint=load_model()
if model is None:
    st.warning("No trained checkpoint found. Run `python scripts/train_tiny.py --steps 1500` from the repository root, then refresh.")
    st.stop()

prompt=st.sidebar.text_area("Prompt","the quick brown fox")
try: ids_list=tokenizer.encode(prompt)
except ValueError as e:
    st.error(str(e)); st.stop()
if not ids_list: st.info("Enter a prompt."); st.stop()
ids_list=ids_list[-model.config.block_size:]
ids=torch.tensor([ids_list])
tokens=list(tokenizer.decode(ids_list))

with torch.no_grad():
    logits,_=model(ids)
    probs=logits[0,-1].softmax(-1)
    vals,inds=probs.topk(min(10,len(probs)))
pred=pd.DataFrame({"next character":[tokenizer.decode([i]) for i in inds.tolist()],"probability":vals.numpy()})

t1,t2,t3,t4,t5,t6,t7=st.tabs(["Prompt Explorer","Attention Explorer","Residual Stream","Neuron Explorer","Intervention Lab","Research Evaluation","Run Artifacts"])
with t1:
    st.subheader("Next-character predictions")
    st.write("TinyGPT is character-level. The built-in corpus is a smoke-test dataset; train on a longer held-out text corpus before interpreting model behaviour.")
    training_info = checkpoint.get("training", {})
    if training_info:
        st.caption(f"Checkpoint corpus: {training_info.get('corpus', 'unknown')} · steps: {training_info.get('steps', 'unknown')} · seed: {checkpoint.get('seed', 'unknown')}")
    st.plotly_chart(px.bar(pred,x="next character",y="probability"),use_container_width=True)
    st.dataframe(pred,use_container_width=True)
with t2:
    st.subheader("Attention patterns")
    layer=st.slider("Layer",0,model.config.n_layer-1,0,key="al")
    heads=collect_attention(model,ids)
    head=st.slider("Head",0,model.config.n_head-1,0,key="ah")
    fig=px.imshow(heads[layer][0,head].numpy(),x=tokens,y=tokens,aspect="auto",
                  labels={"x":"Key token","y":"Query token","color":"Attention"})
    st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    c1.metric("Attention entropy",f"{attention_entropy(heads[layer])[head]:.4f}")
    c2.metric("Mean attention distance",f"{mean_attention_distance(heads[layer])[head]:.4f}")
    st.caption("Attention maps are descriptive and do not establish a head's functional role.")
with t3:
    st.subheader("Residual stream")
    residuals=collect_residuals(model,ids)
    layer=st.slider("Residual checkpoint (0 = embedding)",0,model.config.n_layer,0,key="rl")
    rep=residuals[layer][0]
    norms=torch.linalg.vector_norm(rep,dim=-1).numpy()
    st.plotly_chart(px.line(x=list(range(len(norms))),y=norms,
                  labels={"x":"Token position","y":"L2 norm"}),use_container_width=True)
    coords=pca_projection(rep)
    df=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"token":tokens})
    st.plotly_chart(px.scatter(df,x="PC1",y="PC2",text="token"),use_container_width=True)
    st.caption("PCA is a descriptive projection; 2D geometry may distort the original representation space.")
with t4:
    st.subheader("MLP neuron activations")
    mlps=collect_mlp_hidden(model,ids)
    layer=st.slider("MLP layer",0,model.config.n_layer-1,0,key="ml")
    acts=mlps[layer][0].numpy()
    neuron=st.slider("Neuron index",0,acts.shape[-1]-1,0)
    st.plotly_chart(px.bar(x=tokens,y=acts[:,neuron],labels={"x":"Token","y":"Activation"},
                           title=f"Layer {layer}, neuron {neuron}"),use_container_width=True)
    top=np.argsort(np.abs(acts).mean(0))[-10:][::-1]
    st.dataframe(pd.DataFrame({"neuron":top,"mean_abs_activation":np.abs(acts).mean(0)[top]}))
with t5:
    st.subheader("Causal intervention lab")
    kind=st.selectbox("Component",["Attention head","MLP neuron"])
    layer=st.slider("Intervention layer",0,model.config.n_layer-1,0,key="il")
    maximum=model.config.n_head-1 if kind=="Attention head" else model.config.n_embd*4-1
    component=st.slider("Component index",0,maximum,0)
    if st.button("Run ablation",type="primary"):
        result=run_ablation(model,ids,"head" if kind=="Attention head" else "neuron",layer,component)
        a,b,c=st.columns(3)
        a.metric("Baseline top-token probability",f"{result['baseline_top_probability']:.5f}")
        b.metric("Ablated top-token probability",f"{result['altered_top_probability']:.5f}")
        c.metric("KL divergence",f"{result['kl_baseline_to_altered']:.6f}")
        st.write("Baseline top token:",repr(tokenizer.decode([result["baseline_top_id"]])))
        st.write("Ablated top token:",repr(tokenizer.decode([result["altered_top_id"]])))
        st.warning("Single-prompt result only. Repeat across controlled prompts and report variability before making research claims.")

with t6:
    st.subheader("Research evaluation: multi-prompt causal tests")
    st.write("Run the same intervention over a small prompt suite and report paired descriptive statistics. This is a local demo model, not a pretrained foundation model.")
    default_suite = "the quick brown fox\ntransformers process sequences\na model predicts the next token\nmechanistic interpretability studies internal computation\nattention uses previous context"
    suite_text = st.text_area("One prompt per line", default_suite, height=130)
    suite = []
    for line in suite_text.splitlines():
        line = line.strip()
        if line:
            try:
                suite.append(torch.tensor([tokenizer.encode(line)], dtype=torch.long))
            except ValueError as exc:
                st.warning(f"Skipping unsupported prompt {line!r}: {exc}")
    eval_kind = st.selectbox("Component to evaluate", ["Attention head", "MLP neuron"], key="eval_kind")
    eval_layer = st.number_input("Layer for batch evaluation", 0, model.config.n_layer-1, 0, key="eval_layer")
    max_component = model.config.n_head-1 if eval_kind == "Attention head" else model.config.n_embd*4-1
    eval_component = st.number_input("Component index for batch evaluation", 0, max_component, 0, key="eval_component")
    if st.button("Run multi-prompt evaluation"):
        if len(suite) < 2:
            st.error("Enter at least two valid prompts.")
        else:
            rows = evaluate_ablation_prompts(model, suite, "head" if eval_kind == "Attention head" else "neuron", int(eval_layer), int(eval_component))
            summary = summarize_ablation_rows(rows)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            cols = st.columns(3)
            cols[0].metric("Mean KL divergence", f"{summary['kl_baseline_to_ablated']['mean']:.5f}")
            cols[1].metric("Top-token change rate", f"{summary['top_token_change_rate']['mean']:.1%}")
            cols[2].metric("Prompts evaluated", str(summary['kl_baseline_to_ablated']['n']))
            st.json(summary)
            st.caption("Bootstrap intervals quantify prompt-to-prompt variability; the small prompt suite is exploratory, not a population-level claim.")
            st.download_button("Download per-prompt results CSV", pd.DataFrame(rows).to_csv(index=False), "ablation_results.csv", "text/csv")

    st.divider()
    st.subheader("Component sensitivity sweep")
    st.caption("Ranks components by mean KL change over the supplied prompt set. This is a prioritization tool, not a functional label.")
    sweep_kind = st.selectbox("Sweep component type", ["Attention heads", "MLP neurons"], key="sweep_kind")
    sweep_layer = st.selectbox("Sweep layer", list(range(model.config.n_layer)), key="sweep_layer")
    if st.button("Run component sweep", key="component_sweep_button"):
        if sweep_kind == "Attention heads":
            selected_components = None
            kind_code = "head"
        else:
            activity = collect_mlp_hidden(model, ids)[int(sweep_layer)].abs().mean(dim=(0, 1))
            selected_components = torch.topk(activity, min(20, activity.numel())).indices.tolist()
            kind_code = "neuron"
        sweep_prompts = suite[:min(8, len(suite))]
        if len(sweep_prompts) < 2:
            st.error("Enter at least two valid prompts in the prompt suite above.")
        else:
            detail, summary = sweep_components(model, sweep_prompts, kind_code, int(sweep_layer), selected_components)
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            st.download_button("Download component sweep CSV", pd.DataFrame(detail).to_csv(index=False), "component_sweep.csv", "text/csv")

    st.divider()
    st.subheader("Activation patching demonstration")
    clean_text = st.text_input("Clean prompt", "the quick brown fox")
    corrupt_text = st.text_input("Corrupted prompt (same length recommended)", "the quick brown dog")
    patch_layer = st.slider("Layer to patch", 0, model.config.n_layer-1, 0, key="patch_layer")
    if st.button("Run activation patching"):
        try:
            clean_ids = torch.tensor([tokenizer.encode(clean_text)[-model.config.block_size:]])
            corrupt_ids = torch.tensor([tokenizer.encode(corrupt_text)[-model.config.block_size:]])
            if clean_ids.shape != corrupt_ids.shape:
                st.error("Clean and corrupted prompts must have the same number of characters for aligned patching.")
            else:
                with torch.no_grad():
                    clean_logits, _ = model(clean_ids)
                    target = int(clean_logits[0, -1].argmax())
                patch_result = activation_patch_sweep(model, clean_ids, corrupt_ids, target, position=clean_ids.shape[1]-1)
                st.write("Patched target token:", repr(tokenizer.decode([target])))
                st.json(patch_result)
                patch_df = pd.DataFrame({"layer": list(range(model.config.n_layer)), "target_logit_after_patch": patch_result["patched_target_logits"], "recovery_fraction": patch_result["recovery_fraction"]})
                st.plotly_chart(px.bar(patch_df, x="layer", y="recovery_fraction", title="Target-logit recovery by patched layer"), use_container_width=True)
                st.caption("Recovery fraction is a local intervention metric. Negative or above-1 values are possible and should be interpreted, not clipped.")
        except ValueError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Repeated-token attention probe")
    st.write("Probe whether a repeated token at the end attends to its earlier occurrence. This is a descriptive candidate score, not proof of an induction circuit.")
    repeated_text = st.text_input("Repeated-token sequence", "A B C D A")
    try:
        repeated_ids = torch.tensor([tokenizer.encode(repeated_text)])
        if repeated_ids.shape[1] >= 3:
            scores = induction_score_table(model, [repeated_ids])
            score_df = pd.DataFrame(scores, index=[f"Layer {i}" for i in range(scores.shape[0])], columns=[f"Head {i}" for i in range(scores.shape[1])])
            st.dataframe(score_df.style.format("{:.4f}"), use_container_width=True)
            st.plotly_chart(px.imshow(scores, x=[f"Head {i}" for i in range(scores.shape[1])], y=[f"Layer {i}" for i in range(scores.shape[0])], color_continuous_scale="Viridis", labels={"x":"Head","y":"Layer","color":"Attention mass"}), use_container_width=True)
    except ValueError as exc:
        st.info(f"Enter only characters in the TinyGPT vocabulary: {exc}")

    st.divider()
    st.subheader("Optional pretrained GPT-2 Small workspace")
    st.caption("This uses TransformerLens and downloads GPT-2 weights the first time. It is separate from the locally trained TinyGPT demo.")
    @st.cache_resource(show_spinner="Loading GPT-2 Small; first download may take a while...")
    def load_pretrained_gpt2():
        from transformer_lens import HookedTransformer
        return HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")
    if st.button("Load GPT-2 Small", key="load_gpt2_button"):
        try:
            st.session_state["gpt2_loaded"] = True
            load_pretrained_gpt2()
        except Exception as exc:
            st.session_state["gpt2_loaded"] = False
            st.error(f"Could not load GPT-2 Small: {exc}. Install optional packages with pip install -r requirements-gpt2.txt.")
    if st.session_state.get("gpt2_loaded", False):
        try:
            gpt2 = load_pretrained_gpt2()
            gpt2_prompt = st.text_input("GPT-2 prompt", "The capital of France is", key="gpt2_prompt")
            if st.button("Inspect GPT-2 activations", key="inspect_gpt2_button"):
                gpt2_tokens = gpt2.to_tokens(gpt2_prompt)
                gpt2_logits, gpt2_cache = gpt2.run_with_cache(gpt2_tokens)
                gpt2_probs = gpt2_logits[0, -1].softmax(-1)
                top_probs, top_ids = gpt2_probs.topk(10)
                top_df = pd.DataFrame({"token": [gpt2.to_string(torch.tensor([i])) for i in top_ids.tolist()], "probability": top_probs.detach().cpu().numpy()})
                st.dataframe(top_df, use_container_width=True)
                gpt2_layer = st.slider("GPT-2 layer", 0, gpt2.cfg.n_layers-1, 0, key="gpt2_layer")
                gpt2_head = st.slider("GPT-2 attention head", 0, gpt2.cfg.n_heads-1, 0, key="gpt2_head")
                patt = gpt2_cache[f"blocks.{gpt2_layer}.attn.hook_pattern"][0, gpt2_head].detach().cpu().numpy()
                labels = [gpt2.to_string(torch.tensor([int(t)])) for t in gpt2_tokens[0]]
                st.plotly_chart(px.imshow(patt, x=labels, y=labels, aspect="auto", labels={"x":"Key token", "y":"Query token", "color":"Attention"}), use_container_width=True)
                st.caption("Attention patterns are descriptive. Use the intervention controls below for causal tests.")
            st.markdown("**GPT-2 head ablation**")
            gpt2_ab_layer = st.number_input("Ablation layer", 0, gpt2.cfg.n_layers-1, 0, key="gpt2_ab_layer")
            gpt2_ab_head = st.number_input("Ablation head", 0, gpt2.cfg.n_heads-1, 0, key="gpt2_ab_head")
            if st.button("Ablate GPT-2 head", key="gpt2_ablate_button"):
                tokens = gpt2.to_tokens(gpt2_prompt)
                baseline = gpt2(tokens)[:, -1, :].float()
                def zero_selected_head(z, hook):
                    z = z.clone(); z[:, :, int(gpt2_ab_head), :] = 0; return z
                altered = gpt2.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{int(gpt2_ab_layer)}.attn.hook_z", zero_selected_head)])[:, -1, :].float()
                p = baseline.softmax(-1).clamp_min(1e-12); q = altered.softmax(-1).clamp_min(1e-12)
                st.metric("KL divergence (baseline → ablated)", f"{float((p * (p.log()-q.log())).sum()):.6f}")
                st.write("Baseline next token:", repr(gpt2.to_string(baseline.argmax(-1))))
                st.write("Ablated next token:", repr(gpt2.to_string(altered.argmax(-1))))
        except Exception as exc:
            st.error(f"GPT-2 workspace error: {exc}")


with t7:
    st.subheader("Saved experiment artifacts")
    st.write("This panel reads outputs produced by the command-line experiment scripts. Empty files are not generated automatically; run the research suite first.")
    artifact_dir = ROOT / "artifacts"
    artifact_files = sorted([p for p in artifact_dir.glob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".txt"}])
    if not artifact_files:
        st.info("No research result files found yet. Run `python scripts/run_research_suite.py --prompts 8 --seed 7`.")
    else:
        selected_file = st.selectbox("Select an artifact", artifact_files, format_func=lambda p: p.name)
        if selected_file.suffix.lower() == ".csv":
            artifact_df = pd.read_csv(selected_file)
            st.caption(f"{selected_file.name} · {len(artifact_df):,} rows · {len(artifact_df.columns)} columns")
            st.dataframe(artifact_df, use_container_width=True)
            st.download_button("Download selected CSV", selected_file.read_bytes(), selected_file.name, "text/csv")
        else:
            content = selected_file.read_text(encoding="utf-8")
            try:
                st.json(json.loads(content))
            except json.JSONDecodeError:
                st.code(content)
            st.download_button("Download selected file", content, selected_file.name, "application/json")
