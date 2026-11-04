# Protocol V2: methodological changes

This V2 run protocol should be treated as a new, pre-registered analysis version. Keep all V1 outputs unchanged and do not pool V1 and V2 estimates.

## Changes from V1

1. **Attention target:** The primary induction-style attention score is attention from the final repeated-A query to the earlier successor-B position. Attention to the earlier A remains a descriptive diagnostic only.
2. **MLP discovery score:** Neurons are ranked by the change in their final-query MLP intermediate activation when the earlier B token is replaced by a held-out corruption token. A random neuron control is selected from units with similar mean absolute activation magnitude.
3. **Control task:** An independent, repetition-free random-transition task measures general next-token sensitivity. Discovery, copy evaluation, and control tasks draw from three disjoint token-identity pools.
4. **Activation patching:** V2 excludes the whole-block final-query patch because it can restore the final state almost by construction. It adds (a) whole-block patching at the corrupted successor-B position, (b) a random-position block patch control, (c) selected-vs-control attention-head context patching at the final query, and (d) selected-vs-control MLP-neuron contribution patching at the final query.
5. **Inference:** V1 is exploratory. V2 outputs are a separate analysis set. Report baseline capability and all negative/inconclusive outcomes; do not call an attention head an induction head solely from attention weights.

## Suggested execution

Start with GPT-2 seed 7 using a reduced smoke-test configuration. Inspect the output schema and verify that all V2 patch types exist. Then run GPT-2 seeds 7, 13, and 29 with the full configuration. Repeat the same protocol on SmolLM3-3B-Base in Colab. Pin the same model revision for all seeds within each model.

## Output files

- `discovery_attention_scores_v2.csv`, `discovery_attention_summary_v2.csv`
- `discovery_neuron_ranking_v2.csv`, `component_candidates_v2.csv`
- `heldout_baseline_summary.csv`, `candidate_vs_random_controls.csv`
- `control_candidate_vs_random.csv`, `copy_specificity_summary.csv`
- `activation_patching_per_prompt_v2.csv`, `activation_patching_summary_v2.csv`
- `sequence_copy_cases_v2.csv`, `run_metadata.json`, `RUN_REPORT.md`

The random-transition control is a sensitivity control over random targets, not a benchmark of language modelling ability. Interpret its specificity comparison as a control-task comparison, not proof of a complete circuit.
