# Protocol V1 — exploratory / legacy runs

These outputs belong to the **earlier exploratory protocol** and must not be pooled with Protocol V2. They are preserved for auditability and historical comparison, not presented as corrected-protocol evidence. Classification is based on the archived metadata timestamps and the V1 `method_notes` (final-query interventions and the earlier whole-block patching design), which differ from the explicit Protocol V2 methods.

## Included runs

| Model | Seed | Run timestamp (UTC) | Held-out copy accuracy | Artifact status |
|---|---:|---|---:|---|
| GPT-2 | 7 | 2026-11-03T17:42:54.469388+00:00 | 0.6172 | Full CSV/JSON/Markdown run archive |
| SmolLM3-3B-Base | 7 | 2026-11-03T18:36:35.067951+00:00 | 0.3984 | Full CSV/JSON/Markdown run archive; 14 files |
| SmolLM3-3B-Base | 13 | 2026-11-03T18:42:04.465987+00:00 | 0.4219 | Full CSV/JSON/Markdown run archive; 14 files |
| SmolLM3-3B-Base | 29 | 2026-11-03T18:47:07.451371+00:00 | 0.4062 | Full CSV/JSON/Markdown run archive; 14 files |

A separate SmolLM3 seed-7 smoke test is included at `smollm3/smoke_test_seed_7/`. It used only 16 discovery cases, 24 evaluation cases, and 2 patch cases; it is not one of the three full runs and should not be used as a main result.

## What is included

- Full original CSV/JSON/Markdown outputs expanded into model/seed folders.
- Original uploaded result ZIPs preserved unchanged in `raw_archives/`.
- V1 run reports, per-prompt data, discovery rankings, candidate/control comparisons, and metadata where present.

## Interpretation

The V1 numbers describe the earlier protocol and should be reported separately. A high baseline or selected component in V1 is not a validation of the corrected V2 method. For V2 use `../protocol_v2/`.
