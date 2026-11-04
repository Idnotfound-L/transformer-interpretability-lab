# Transformer Internals & Mechanistic Interpretability Lab

A research-engineering project for inspecting and causally probing decoder-only Transformer language models. The repository combines a PyTorch Transformer implementation, Hugging Face model instrumentation, a controlled sequence-copying study (Protocol V2), ablation and activation-patching experiments, statistical summaries, and an offline Streamlit dashboard.

**Models studied:** GPT-2 (`gpt2`) and SmolLM3-3B-Base (`HuggingFaceTB/SmolLM3-3B-Base`).

**Primary research question:** Which attention heads and MLP coordinates causally contribute to a controlled repeated-token sequence-copying behaviour, and how robust are these effects across independent seeds and model families?

> **Research integrity.** The results folder contains full raw V2 output bundles (CSV, JSON, report) for GPT-2 and SmolLM3-3B-Base, seeds 7, 13 and 29, and full raw V1 exploratory bundles kept separately. Headline figures were cross-checked against the raw CSVs. Inference is within-seed; no pooled test across seeds has been run. The SmolLM3 Layer 15 Head 11 result is described as a candidate that causally contributes to the synthetic task, not as a proven complete induction circuit. See [`results/README.md`](results/README.md) and [`reports/FINAL_RESEARCH_REPORT.md`](reports/FINAL_RESEARCH_REPORT.md).

## What the project implements

- **Architecture-aware Hugging Face backend:** GPT-2-style and Llama-like decoder stacks, eager attention inspection, model metadata and GQA-aware hooks.
- **Protocol V2 discovery:** final-query attention lift to the earlier successor token B; earlier-A attention is retained as a diagnostic. MLP coordinates are ranked by clean-versus-corrupted final-query activation sensitivity.
- **Causal ablation:** selected attention heads and MLP coordinates are compared against same-layer random heads and activity-matched random neurons on held-out copy prompts.
- **Task-specificity control:** the same components are evaluated on a repetition-free random-transition control task.
- **Activation patching:** clean block-state patches at the earlier successor position, random-position controls, and component-specific head/neuron patches. The trivial whole-block final-query patch is omitted.
- **Statistics:** per-prompt metrics, bootstrap confidence intervals, paired sign-flip tests and Benjamini–Hochberg adjusted q-values.
- **Dashboard:** saved Protocol V2 runs can be inspected without reloading model weights.
- **Reproducibility:** seed, model revision, hardware, precision, selected layers, prompt hashes and experiment settings are exported in JSON metadata.

## Repository layout

```text
transformer-interpretability-lab/
├── src/interpretability_lab/     # model, hooks, experiment logic, Protocol V2
├── scripts/                     # training, research runners, V2 run indexer
├── dashboard/                   # TinyGPT, Hugging Face and Protocol V2 dashboards
├── tests/                       # automated tests
├── data/                        # small prompt examples/documentation
├── docs/                        # architecture, method, reproducibility notes
├── results/                     # available run reports and provenance notes
├── reports/                     # final research report
└── artifacts/                   # local experiment outputs (git-ignored by default)
```

## Setup

Python 3.11 is recommended. On Windows:

1. Extract this ZIP to a short path, e.g. `C:\mi\` (helps avoid Windows long-path installation issues).
2. Run `setup_windows.bat`.
3. Run `setup_huggingface_windows.bat` for pretrained model experiments.
4. Verify the project with `\.venv\Scripts\python.exe -m pytest -q`.

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-huggingface.txt
pip install -e .
```

The test suite currently passes **28 tests** in the packaging environment. Tests exercise software components and synthetic model adapters; passing tests is not a substitute for empirical validation on pretrained checkpoints.

## Run Protocol V2

### GPT-2 on CPU or GPU

Windows quick runner (defaults to seed 7):

```powershell
.\run_protocol_v2_gpt2_windows.bat 7
```

Run individual seeds directly on any OS:

```bash
python scripts/run_sequence_copy_study_v2.py \
  --model-id gpt2 --device auto --dtype auto \
  --discovery-cases 96 --evaluation-cases 128 --control-cases 128 \
  --patch-cases 12 --layers 8 --batch-size 2 --seed 7
```

Repeat with `--seed 13` and `--seed 29`. Keep each timestamped output folder. V2 writes to `artifacts/sequence_copy_v2/` by default and does not overwrite Protocol V1 outputs.

### SmolLM3-3B-Base on a cloud GPU

Use a CUDA GPU with sufficient memory (the user's completed runs used an NVIDIA T4 in BF16). Install Transformers 4.53+ and required Hugging Face packages without reinstalling a preconfigured CUDA PyTorch build:

```bash
pip install "transformers>=4.53.0,<5" accelerate safetensors sentencepiece pandas tabulate
```

Run one seed:

```bash
python scripts/run_sequence_copy_study_v2.py \
  --model-id HuggingFaceTB/SmolLM3-3B-Base \
  --revision d78a42f79198603e614095753484a04c10c2b940 \
  --device cuda --dtype auto \
  --discovery-cases 96 --evaluation-cases 128 --control-cases 128 \
  --patch-cases 12 --layers 8 --batch-size 1 --seed 7 \
  --output-dir /path/to/persistent/storage/smollm3_protocol_v2
```

Repeat for seeds 13 and 29, preserving the same model revision and protocol settings. On a T4, batch size 1 is the conservative setting; batch size 4 may reduce runtime if memory permits. Save outputs to persistent storage (Google Drive or download from Kaggle) before the runtime ends.

### Output files

Each Protocol V2 run folder should contain:

| File | Purpose |
|---|---|
| `RUN_REPORT.md` | Human-readable per-run report |
| `run_metadata.json` | Protocol, model, hardware, precision, seeds, hashes and settings |
| `sequence_copy_cases_v2.csv` | Exact discovery/evaluation/control prompts and token IDs |
| `discovery_attention_scores_v2.csv` / `discovery_attention_summary_v2.csv` | Successor-B attention scoring |
| `discovery_neuron_ranking_v2.csv` | Corruption-sensitive MLP neuron ranking |
| `component_candidates_v2.csv` | Discovery-selected components and matched controls |
| `heldout_baseline_metrics.csv` / `heldout_baseline_summary.csv` | Baseline task performance and uncertainty |
| `candidate_vs_random_controls.csv` | Selected-versus-random copy-task ablation effects |
| `control_candidate_vs_random.csv` | Same comparison on random-transition control |
| `copy_specificity_summary.csv` | Copy effect minus control effect |
| `activation_patching_per_prompt_v2.csv` / `activation_patching_summary_v2.csv` | Block-position and component-specific patching |

## Dashboard

After running experiments, launch the Protocol V2 offline dashboard:

```bash
streamlit run dashboard/protocol_v2_dashboard.py
```

On Windows, double-click `run_protocol_v2_dashboard_windows.bat`. Set the dashboard run directory to the folder containing the timestamped run folders if it is not the default `artifacts/sequence_copy_v2`.

The dashboard reads saved CSV/JSON outputs only. It does not download model weights or rerun experiments.

## Cross-seed indexing

To index all Protocol V2 run folders under a directory:

```bash
python scripts/summarize_v2_runs.py artifacts/sequence_copy_v2
```

For SmolLM3, point the utility at the persistent parent directory containing all three seed folders. It writes `run_index.csv`, per-seed tables and a descriptive component-consistency table. The utility does not pool p-values or replace a prompt-level meta-analysis.

## Research findings and limitations

See:
- [`reports/FINAL_RESEARCH_REPORT.md`](reports/FINAL_RESEARCH_REPORT.md)
- [`results/README.md`](results/README.md)
- [`results/smollm3_protocol_v2/THREE_SEED_FINDINGS_AUTHOR_SUMMARY.md`](results/smollm3_protocol_v2/THREE_SEED_FINDINGS_AUTHOR_SUMMARY.md)

Key limitations to communicate in interviews:

1. The probe is a synthetic sequence-copying task, not a natural-language benchmark.
2. SmolLM3's baseline performance is partial; negative copy margins and low target probability constrain mechanistic claims.
3. Attention weights are candidate-discovery signals, not functional proof.
4. Random-transition accuracy is at 0%, so the control may have a floor effect.
5. Only eight of 36 layers and one top candidate per layer were screened in the reported protocol.
6. Component-specific patching was weak, and the block-state patching set was small.
7. Ablation can be off-manifold; avoid claiming that one component is a complete circuit.
8. Cross-model layer/head indices are not directly comparable.

## CV-ready wording

Use the evidence-backed entry in [`CV_ADDITION.txt`](CV_ADDITION.txt). The strongest result is worded as a reproducible causal contribution on a synthetic task, not as discovery of a complete induction circuit.
