# Project scope and CV guidance

## What the repository implements

- A decoder-only GPT-style Transformer in PyTorch with tied embeddings, causal self-attention, residual blocks and MLPs.
- Forward-hook instrumentation and explicit capture of attention patterns, residual stream states and MLP activations.
- Interactive Streamlit views for next-token predictions, attention maps, residual representations, neuron activity and causal interventions.
- Attention-head and MLP-neuron ablation, multi-prompt component sweeps, KL / prediction-change metrics and prompt-level bootstrap intervals.
- Position-aligned residual activation patching with clean / corrupted baselines and a corrupted-self-patch control.
- An optional GPT-2 Small + TransformerLens CLI pipeline for full head sweeps, activity-selected neuron ablations, repeated-token probes and patching.
- Checkpointed training/validation metrics, seed handling, prompt hashes, environment metadata, CSV/JSON exports, tests and CI configuration.

## What still requires execution and interpretation

- Train TinyGPT on a suitable text corpus and inspect the held-out validation loss.
- Install optional TransformerLens dependencies and download GPT-2 Small weights.
- Run the GPT-2 research suite on hardware with realistic runtime expectations.
- Use separate discovery and evaluation prompt files for confirmatory MLP-neuron tests.
- Check tokenization and position alignment for every patching pair.
- Inspect all exported per-prompt rows, controls, confidence intervals and null results.
- Write a research note that distinguishes measured observations from proposed mechanisms.

## Implementation-focused CV entry

**Transformer Internals & Mechanistic Interpretability Lab | PyTorch, TransformerLens, Streamlit**

- Built a decoder-only Transformer in PyTorch and instrumented attention, residual-stream and MLP activations with forward hooks; developed an interactive dashboard for layer/head/neuron-level inspection.
- Implemented causal attention-head and MLP-neuron ablations plus layer-wise activation patching, with multi-prompt KL-divergence evaluation, bootstrap confidence intervals, controls and experiment exports.
- Developed a reproducible GPT-2 Small analysis pipeline using TransformerLens, including repeated-token attention probes, prompt-set hashing and model/runtime metadata capture.

Use only bullets that accurately describe completed work. The code is present in the repository, but pretrained GPT-2 experiments have not been executed as part of packaging. Add quantitative outcomes only after running them and preserving the generated CSV/JSON artifacts.

## Hugging Face research extension

The repository now also includes an architecture-aware Hugging Face adapter for decoder-only causal LMs, defaulting to SmolLM3-3B-Base. It supports GQA-aware query-head ablation through the attention output projection, gated MLP-coordinate ablation through the down projection, layer-wise block-output activation patching, repeated-token probes and same-prompt cross-model aggregate comparisons. The HF runner separates MLP discovery prompts from evaluation prompts and records model configuration, precision, device, prompt hashes, per-prompt metrics and bootstrap intervals.

The model weights are not bundled. The first pretrained run needs network access to fetch the Hugging Face checkpoint and hardware with enough memory. For a bounded first run, evaluate two layers and four query heads per layer; expand after checking runtime and memory. Refer to `docs/HUGGINGFACE_RESEARCH.md` for exact commands, interpretation caveats and output definitions.

## Updated implementation-focused CV entry

**Transformer Internals & Mechanistic Interpretability Lab | PyTorch, Hugging Face Transformers, Streamlit**

- Built an architecture-aware instrumentation framework for pretrained decoder-only Transformers, using PyTorch hooks to inspect attention projections and gated MLP activations across Hugging Face models.
- Implemented GQA-aware query-head ablation, MLP-neuron interventions and layer-wise activation patching; evaluated prediction changes using KL divergence, target-logit recovery and prompt-level bootstrap intervals.
- Developed a Streamlit research dashboard and reproducible experiment pipeline with discovery/evaluation prompt separation, metadata capture, and cross-model comparison support for GPT-2 and SmolLM3-3B.

These bullets describe implemented software, not completed scientific discoveries. Add quantified results only after you run the pretrained models and verify the CSV/JSON outputs.


## Primary controlled sequence-copying study

The main research track is documented in `docs/SEQUENCE_COPY_STUDY.md` and implemented by `scripts/run_sequence_copy_study.py`. It creates fixed-length repeated-token copy prompts, allocates distinct token identities to discovery and held-out evaluation cases, ranks attention heads and MLP coordinates on discovery data, then performs final-query ablations against same-layer random controls. Paired candidate-control effects are reported with bootstrap confidence intervals, sign-flip permutation tests and Benjamini-Hochberg q-values. Activation patching corrupts the earlier successor token and restores clean block states at that token position or at the final query position.

The study is deliberately designed to be falsifiable. If the model has poor baseline copy accuracy, or discovery-ranked components do not outperform random controls, the report should say so. Synthetic copying is a narrow behavioural probe and cannot establish a general semantic role for a head. Cross-model validation is planned for GPT-2 and SmolLM3-3B-Base; model-specific results must be interpreted separately.
