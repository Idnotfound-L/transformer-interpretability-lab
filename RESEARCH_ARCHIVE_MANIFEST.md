# Complete research archive manifest

## Archive purpose

This package contains the project codebase, dashboard, tests, documentation, and the run artifacts that were available at packaging time. Experiment results are organized into separate Protocol V1 and Protocol V2 trees to prevent accidental pooling.

## Protocol inventory

| Protocol | Model | Runs represented | Raw data availability |
|---|---|---|---|
| V1 (exploratory) | GPT-2 | seed 7 | Full CSV/JSON/Markdown files and original ZIP |
| V1 (exploratory) | SmolLM3-3B-Base | seeds 7, 13, 29 | Full CSV/JSON/Markdown files and original ZIP |
| V1 (smoke test) | SmolLM3-3B-Base | seed 7 | Included separately; 16 discovery / 24 evaluation / 2 patch cases |
| V2 (corrected) | GPT-2 | seeds 7, 13, 29 | Full per-seed CSV/JSON/Markdown bundles (19 files each) in `results/protocol_v2/gpt2/seed_*` |
| V2 (corrected) | SmolLM3-3B-Base | seeds 7, 13, 29 | Full per-seed CSV/JSON/Markdown bundles (19 files each) in `results/protocol_v2/smollm3/seed_*`, plus the consolidated author summary |

## Main research caution

The V1 run reports are not V2 results. Do not combine their CSV rows, means, confidence intervals, or significance tests with the corrected V2 results. The corrected V2 reports and available V2 metadata are under `results/protocol_v2/`.

## Code and reproducibility

- V1 sequence-copy implementation: `src/interpretability_lab/sequence_copy.py` and the corresponding legacy runner.
- V2 corrected implementation: `src/interpretability_lab/sequence_copy_v2.py`, `scripts/run_sequence_copy_study_v2.py`, and the V2 dashboard.
- Protocol documentation: `docs/EXPERIMENT_PROTOCOL.md`, `docs/PROTOCOL_V2_CHANGELOG.md`, `docs/REPRODUCIBILITY_CHECKLIST.md`.
- The original project package recorded 28 passing tests at packaging time; this archive reorganization itself does not change source code.

## Evidence limitations

- V2 GPT-2 runs used CPU/float32 and did not record a resolved model revision, so the exact GPT-2 weights are not pinned in the metadata.
- V2 SmolLM3 raw per-seed outputs are included. The headline figures in the consolidated summary (held-out accuracy, L15 H11 margin drop and specificity, BH q-values, patching pattern) were cross-checked against these raw CSVs; no formal pooled or meta-analytic test has been run across seeds.
- The random-transition control reported in V2 has 0% top-1 accuracy and may be a floor-effect control.
- The task is a synthetic sequence-copying probe. It does not prove a complete induction circuit or a natural-language mechanism.

## Integrity

A file-level SHA-256 inventory is provided in `ARCHIVE_SHA256SUMS.txt`. Original V1 result ZIPs are retained in `results/protocol_v1/raw_archives/`.
