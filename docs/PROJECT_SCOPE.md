# Project scope

## What the repository implements

- A decoder-only GPT-style Transformer in PyTorch (tied embeddings, causal self-attention, residual blocks, MLPs) with a small training pipeline.
- Forward-hook instrumentation that captures attention patterns, residual-stream states and MLP activations.
- Interactive Streamlit views for next-token predictions, attention maps, residual representations, neuron activity and causal interventions.
- Attention-head and MLP-neuron ablation, multi-prompt component sweeps, KL / prediction-change metrics and prompt-level bootstrap intervals.
- Position-aligned residual activation patching with clean / corrupted baselines and a corrupted-self-patch control.
- An optional GPT-2 Small + TransformerLens CLI pipeline for full head sweeps, activity-selected neuron ablations, repeated-token probes and patching.
- An architecture-aware Hugging Face adapter for decoder-only causal LMs (GPT-2-style and Llama-like stacks, default SmolLM3-3B-Base) with GQA-aware query-head ablation through the attention output projection, gated MLP-coordinate ablation through the down projection, layer-wise block-output patching and cross-model aggregate comparisons. See [`HUGGINGFACE_RESEARCH.md`](HUGGINGFACE_RESEARCH.md).
- Per-run metadata capture: seed, model revision, hardware, precision, selected layers and prompt hashes.

## Primary study: controlled sequence copying

The main research track is the Protocol V2 sequence-copying study ([`SEQUENCE_COPY_STUDY.md`](SEQUENCE_COPY_STUDY.md), [`PROTOCOL_V2_CHANGELOG.md`](PROTOCOL_V2_CHANGELOG.md)), implemented in `src/interpretability_lab/sequence_copy_v2.py` and `scripts/run_sequence_copy_study_v2.py`.

It builds fixed-length repeated-token prompts, gives discovery and held-out evaluation cases disjoint token pools, ranks attention heads and MLP coordinates on discovery data, and ablates them against same-layer random controls on held-out prompts. Paired candidate-versus-control effects are reported with bootstrap confidence intervals, sign-flip permutation tests and Benjamini–Hochberg q-values. Activation patching corrupts the earlier successor token and restores clean block states at that position.

The study is designed to be falsifiable: if baseline copy accuracy is poor, or discovery-ranked components do not beat random controls, the reports say so. Synthetic copying is a narrow behavioural probe and cannot establish a general semantic role for a head. Results for GPT-2 and SmolLM3-3B-Base are interpreted separately; layer/head/neuron indices are not comparable across architectures.

## Out of scope / future work

- Natural-language induction benchmarks with a difficulty-matched control task.
- Screening more than one candidate per layer, and more than 8 of the model's layers.
- Combined ablations and path patching from candidate heads (e.g. SmolLM3 L15 H11) to downstream components.
- More patching prompts, and a pre-specified prompt-level meta-analysis across seeds.
