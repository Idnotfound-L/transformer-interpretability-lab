# Architecture

## Execution paths

**Local TinyGPT path**: `train_tiny.py` loads the selected text corpus, builds a character vocabulary, performs a chronological train/validation split, trains the decoder-only model and writes a checkpoint plus loss history. `dashboard/app.py` reloads this checkpoint and uses the model's saved vocabulary. `run_research_suite.py` evaluates all attention heads and a selected set of activity-ranked MLP neurons, performs repeated-token probes and runs position-aligned activation patching.

**Pretrained GPT-2 path**: `run_gpt2_research.py` loads GPT-2 Small through TransformerLens. It runs head ablations over a prompt set, discovers high-activity MLP neurons and tests their causal sensitivity, probes repeated-token attention, and patches clean residual activations into a corrupted run. Every prompt-level metric is preserved before aggregation.

## Intervention semantics

- **Head ablation** zeros one head's output at the attention `hook_z` / pre-output-projection representation. This tests sensitivity to that head's contribution, not what semantic feature the head represents.
- **Neuron ablation** zeros one MLP post-activation dimension. This is a feature-coordinate intervention; individual neuron effects can be polysemantic and context-dependent.
- **Activation patching** copies one aligned token-position residual activation from a clean run into a corrupted run at a specified block output. The final next-token logit is tracked through later blocks. Recovery is not clipped to [0, 1].
- **Repeated-token probes** measure attention from the later occurrence of a repeated token to its earlier occurrence. This is a descriptive screening heuristic, not a verified induction circuit.

## Metrics

- `KL(p_baseline || p_intervened)`: distribution change under an intervention.
- `top_token_changed`: whether the argmax token changes.
- `baseline_top_logit_change`: change to the baseline argmax token's logit.
- `logit_recovery_fraction`: `(patched_logit - corrupted_logit) / (clean_logit - corrupted_logit)`.
- Prompt bootstrap intervals: resample prompt-level observations for a given component; intervals characterize uncertainty over the sampled prompt set only.

## Separation of concerns

`core.py` owns the model and local interventions. `research.py` owns local evaluation and patching protocols. `experiment.py` owns seed and run-provenance utilities. CLI scripts orchestrate experiments and serialize artifacts. The dashboard visualizes existing model state and experiment outputs; it does not silently write or mutate checkpoints.

## Hugging Face pretrained-model path

**Pretrained HF path**: `run_hf_research.py` loads a causal language model through `AutoModelForCausalLM`. `hf_backend.py` resolves a decoder stack and the attention/MLP output projections for Llama-like grouped-query models and GPT-2-style blocks. The experiment runner uses discovery prompts to rank MLP coordinates, then measures their ablation effects on held-out evaluation prompts. It exports per-prompt results, bootstrap summaries, repeated-token attention probes, layer-wise block-output patching, prompt hashes and model/environment metadata.

### Projection hook semantics

- **GQA-aware query-head ablation** hooks the input to the attention output projection. For SmolLM3, this is a concatenation of 16 query-head context vectors even though only four key/value heads are shared. Zeroing one `head_dim` slice intervenes on a query head, not a KV group.
- **Gated MLP-coordinate ablation** hooks the input to `down_proj`, which receives the post-gating intermediate representation. One intermediate coordinate is zeroed before output projection.
- **Layer-wise patching** captures a decoder-block output from a clean prompt, then replaces the final-token output at that block during a corrupted-prompt pass. Same tokenized lengths are required.
- **Attention patterns** are requested through eager attention when supported. They are exploratory descriptors; causal interventions and controls are needed to support functional hypotheses.

`dashboard/hf_app.py` is a separate entry point so the original TinyGPT dashboard continues to work without loading a multi-billion-parameter model.
