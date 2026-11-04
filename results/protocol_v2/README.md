# Protocol V2 — corrected protocol

Protocol V2 is the corrected study design. See `../../docs/PROTOCOL_V2_CHANGELOG.md` and `../../docs/EXPERIMENT_PROTOCOL.md` for the method changes. These results are kept separate from the older exploratory V1 runs.

## GPT-2

| Seed | Held-out copy accuracy | Included artifacts |
|---:|---:|---|
| 7 | 0.4375 | Full raw bundle (19 files) |
| 13 | 0.5000 | Full raw bundle (19 files) |
| 29 | 0.5391 | Full raw bundle (19 files) |

All three GPT-2 V2 seeds include full raw bundles, and the reports match the raw CSVs. Runs used CPU/float32 and the model revision was not recorded.

## SmolLM3-3B-Base

`smollm3/seed_7`, `seed_13` and `seed_29` contain the full raw V2 output bundles (CSV, JSON and run report). `smollm3/THREE_SEED_FINDINGS_AUTHOR_SUMMARY.md` is the consolidated summary; its headline figures were cross-checked against the raw CSVs. Per-seed inference is within-seed (BH-corrected); no pooled test across seeds has been run.

## Important separation rule

Do not copy V1 CSVs into this folder or combine them with V2 summary statistics. The V1 data are included under `../protocol_v1/` only as historical exploratory runs.
