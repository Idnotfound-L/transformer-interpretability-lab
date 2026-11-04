# Reproducibility and research-integrity checklist

Complete this checklist before using numerical findings in a CV, report, presentation or interview.

## Before running

- [ ] Record the exact model ID and pin a model revision/commit when possible.
- [ ] Record the tokenizer and Transformers/PyTorch versions.
- [ ] Use a fixed seed and preserve the exact command.
- [ ] Preserve `sequence_copy_cases.csv`, including input token IDs and decoded prompts.
- [ ] Verify discovery and evaluation token pools are disjoint.
- [ ] Keep all sampled token identities unique across prompts within each split.
- [ ] Do not change the evaluation prompts after inspecting candidate effects.

## During evaluation

- [ ] Report held-out baseline copy accuracy and its confidence interval first.
- [ ] Report all discovery-ranked candidate components, not only the largest effect.
- [ ] Include same-layer random component controls.
- [ ] Keep the intervention position fixed at the final query token for ablation.
- [ ] Preserve every prompt-level result, including null, negative and inconsistent effects.
- [ ] Report confidence intervals, permutation p-values and BH-adjusted q-values.
- [ ] Inspect activation-patching results at both the corrupted successor and final query positions.
- [ ] Repeat the analysis with at least three random seeds before making strong generalization claims.

## Before making a mechanistic claim

- [ ] Baseline model behaviour supports the task being studied.
- [ ] Candidate selection and causal evaluation use separate prompt/token pools.
- [ ] Candidate effects are distinguishable from random-control effects.
- [ ] Evidence is consistent across ablation and activation patching, or disagreements are explained.
- [ ] Attention maps are not treated as causal evidence by themselves.
- [ ] Potential off-manifold effects of ablation are acknowledged.
- [ ] Synthetic sequence-copying findings are not generalized to natural language without further tests.
- [ ] Cross-model comparisons do not assume matching head indices imply matching functions.

## Before publishing

- [ ] Preserve raw CSV/JSON outputs, run metadata and the generated report.
- [ ] Include software/hardware environment and any deviations from protocol.
- [ ] Include null results and limitations.
- [ ] Use only measured values in CV bullets; do not copy example placeholders.
