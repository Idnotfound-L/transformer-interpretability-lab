# Hugging Face research workflow

## Primary checkpoint

The default checkpoint is `HuggingFaceTB/SmolLM3-3B-Base`, a 3B-parameter decoder-only base model. SmolLM3 uses 36 decoder layers, 16 query heads and 4 key/value heads (grouped-query attention), a hidden size of 2,048 and a SwiGLU-style intermediate width of 11,008. The official Transformers integration is available from Transformers 4.53.0 onward. See the [model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base), [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/smollm3), and [Hugging Face technical blog](https://huggingface.co/blog/smollm3).

Weights are not included in this repository. They are downloaded from Hugging Face the first time a model is loaded. The base checkpoint is used rather than an instruction-tuned checkpoint so the experiment starts from a pretrained base language model.

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-huggingface.txt
pip install -e .
```

## First bounded research run

```bash
python scripts/run_hf_research.py \
  --model-id HuggingFaceTB/SmolLM3-3B-Base \
  --device auto \
  --layers 2 \
  --max-heads 4 \
  --neuron-top-k 2
```

For strict reproducibility, pin a Hugging Face commit or tag using `--revision REVISION`. The resolved checkpoint commit hash is written to `run_metadata.json`.

The defaults evaluate the first two layers and the first four query heads per layer to keep the first run bounded. After confirming memory and runtime, increase `--layers` and `--max-heads` gradually. The runner also selects MLP neurons using a discovery prompt file, then evaluates their ablation effects on a separate evaluation prompt file. Standard tokenizer special tokens are added where the tokenizer defines them (for example, a beginning-of-sequence token). The runner warns and records exact prompt overlap between discovery and evaluation sets.

Use the same evaluation prompt file for model comparisons:

```bash
python scripts/run_hf_research.py --model-id gpt2 --evaluation-prompts data/evaluation_prompts.txt --discovery-prompts data/discovery_prompts.txt --layers 4 --max-heads 4
python scripts/run_hf_research.py --model-id HuggingFaceTB/SmolLM3-3B-Base --evaluation-prompts data/evaluation_prompts.txt --discovery-prompts data/discovery_prompts.txt --layers 4 --max-heads 4
python scripts/compare_model_runs.py artifacts/hf_gpt2_TIMESTAMP artifacts/hf_HuggingFaceTB_SmolLM3-3B-Base_TIMESTAMP
```

The comparison utility checks that evaluation-prompt hashes match. It compares aggregate intervention sensitivity; it does **not** assume that layer or head indices have the same function across architectures.

## Dashboard

```bash
streamlit run dashboard/hf_app.py
```

The dashboard can load a Hugging Face causal LM, inspect next-token probabilities and attention maps, run single-prompt head/neuron ablations, perform layer-wise activation patching, and browse saved CLI results.

## What the interventions mean

### Attention-head ablation (GQA-aware)

The adapter places a forward pre-hook on each layer's attention output projection (`o_proj` for Llama-like/GQA models). That projection receives the concatenation of query-head context vectors. It zeros the selected `head_dim` slice immediately before the output projection. For SmolLM3, this addresses one of 16 query heads even though four key/value heads are shared across query-head groups. It is not an ablation of an entire key/value group.

### MLP-neuron ablation

For gated MLPs, the adapter hooks the input to `down_proj`, where the gated intermediate vector is present, then zeros one intermediate coordinate. For GPT-2-style MLPs, it hooks the corresponding `c_proj` input. This is a coordinate intervention, not proof that the unit has a stable semantic meaning.

### Activation patching

The runner replaces the final-token hidden state at one decoder-block output with the corresponding clean-prompt activation, then measures target-logit recovery and KL divergence. Clean and corrupted prompts must tokenize to the same sequence length. The self-patch control should reproduce the corrupted baseline; inspect per-layer outputs rather than interpreting recovery as a probability or a percentage guaranteed to lie between zero and one.

### Attention probe

Repeated-token prompts test whether the final repeated token attends to an earlier matching occurrence. This is a descriptive diagnostic. A high attention score alone does not establish an induction circuit or a causal behaviour.

## Outputs

Every run creates a timestamped folder under `artifacts/` containing:

- `ablation_per_prompt.csv`: all prompt/component outcomes;
- `ablation_summary.csv`: per-component means and prompt-bootstrap intervals;
- `repeated_token_attention.csv`: per-layer/head repeated-token matching scores;
- `activation_patching.csv` or a `SKIPPED` reason;
- `selected_neurons.json`: discovery-selected MLP coordinates;
- `run_metadata.json`: model configuration, model ID, precision, software/hardware context, seeds, prompt hashes and caveats;
- `RUN_REPORT.md`: generated summary of measured run outputs.

The runner records values from the actual model at execution time. No performance numbers or scientific conclusions are bundled as claimed findings.

## Hardware and runtime

A 3B checkpoint in BF16/FP16 requires approximately 6 GB for weights alone; runtime memory also includes temporary activations, attention maps and framework overhead. FP32 CPU loading needs substantially more RAM. Use short prompts, keep `--layers` small for the first run, and consider a GPU with adequate VRAM. The repository currently uses full-precision model weights (BF16/FP16/FP32 selection), not 4-bit quantization. If you cannot load the 3B model, validate the workflow with `gpt2` first or use an appropriate smaller checkpoint.

## Research discipline

- Treat discovery prompts as a screening set and evaluation prompts as held-out evidence.
- Keep evaluation prompts fixed across model comparisons and verify matching prompt hashes.
- Report prompt-level observations, confidence intervals and null effects.
- Attention is descriptive; combine it with intervention and patching evidence.
- Ablation can push activations off the model's normal distribution. Triangulate with patching, matched controls and alternative prompts.
- Do not write a CV claim about a discovered circuit or a measured effect until you have inspected and understood the saved output files.
