# Project status — final bundle

## Implemented

- PyTorch decoder-only Transformer and training/demo pipeline.
- Architecture-aware Hugging Face decoder adapter for supported GPT-2-like and Llama-like stacks.
- GQA-aware attention query-head ablation and post-gating MLP-coordinate ablation.
- Protocol V2 repeated-token sequence-copying experiment with three disjoint token pools.
- Successor-B attention discovery scoring and corruption-sensitive MLP neuron ranking.
- Held-out candidate ablations with same-layer random-head and activity-matched neuron controls.
- Repetition-free random-transition control-task evaluation and copy-specificity comparison.
- Whole-block position patching and component-specific head/neuron patching.
- Prompt-level metrics, bootstrap confidence intervals, paired sign-flip tests and BH adjustment.
- Per-run CSV/JSON/Markdown exports and offline Protocol V2 Streamlit dashboard.
- Cross-seed indexer for descriptive aggregation.

## Verification in this packaging environment

- `python -m pytest -q`: **28 passed**.
- Python compilation checks are run during packaging for scripts, source modules and dashboards.

## Empirical results included

- Protocol V1 exploratory outputs: complete raw CSV/JSON/Markdown files for GPT-2 seed 7 and SmolLM3 seeds 7, 13 and 29, plus a separate SmolLM3 smoke test.
- GPT-2 Protocol V2: full raw bundles (19 files each) for seeds 7, 13 and 29.
- SmolLM3-3B Protocol V2: consolidated three-seed findings summary supplied by the project author.
- V1 and V2 are stored in separate trees under `results/protocol_v1/` and `results/protocol_v2/`.

## Important evidence limitations

- All V2 raw bundles (GPT-2 and SmolLM3, seeds 7, 13, 29) are included and the headline figures were cross-checked against them. GPT-2 runs ran on CPU in float32 with the model revision unpinned (`resolved_revision` is null).
- V1 raw data are included for historical reproducibility, but must not be pooled with V2 results.
- The six runs' summary claims should not be confused with a pooled inferential test. A formal seed-level meta-analysis would require full per-prompt data and should be pre-specified.
- The random-transition control is a generic sensitivity control and had 0% accuracy in reported runs; consider a difficulty-matched natural-text control in future work.
- The current study is a synthetic probe. It does not establish a complete induction circuit or broad natural-language mechanism.

## CV guidance

The CV entry uses a qualified finding: a candidate component makes a reproducible causal contribution to the measured synthetic task. Do not call it a complete induction head/circuit.
