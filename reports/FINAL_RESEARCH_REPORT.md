# Research report: causal contributions to synthetic sequence copying in GPT-2 and SmolLM3-3B

## Abstract

This project builds and applies a Protocol V2 mechanistic-interpretability workflow to synthetic repeated-token sequence copying in GPT-2 and SmolLM3-3B-Base. The workflow uses disjoint discovery, held-out copy-evaluation and random-transition control token pools; successor-token attention scoring; corruption-sensitive MLP-neuron ranking; component ablation against matched random controls; prompt-level bootstrap intervals; paired sign-flip tests with Benjamini–Hochberg correction; and activation patching. Three seed runs were completed for each model; raw bundles are in `results/protocol_v2/`. The most reproducible reported SmolLM3 result is a discovery-selected Layer 15, Head 11 whose ablation has a larger effect on copy prompts than matched random-head controls across all three seeds. This is evidence for a causal contribution on the synthetic task, not proof of a complete induction circuit.

## Research question

Which attention heads and MLP coordinates causally contribute to a controlled repeated-token copying behaviour, and how consistent are those effects across GPT-2 and SmolLM3-3B-Base?

The synthetic task has the form `[prefix, A, B, fillers, A] → predict B`. Discovery and evaluation prompts are separated by token identity pools, so a component is selected using discovery data and evaluated on held-out prompts.

## Protocol V2 methods

1. **Discovery:** rank attention heads by final-query attention lift to the earlier successor token B. Attention to the earlier A is retained as a diagnostic. Rank MLP coordinates by activation sensitivity to corruption of B.
2. **Controls:** pair selected heads with same-layer random heads and selected neurons with activity-matched random neuron coordinates.
3. **Causal evaluation:** ablate candidate and control components at the final query on held-out copy prompts and on an independent repetition-free random-transition control task.
4. **Statistics:** compute prompt-level margin effects, bootstrap confidence intervals, paired sign-flip tests and BH-adjusted q-values within each run.
5. **Activation patching:** patch clean block states at the earlier successor position, use random-position controls, and perform component-specific head/neuron patches. The trivial whole-block final-query patch is omitted.

## GPT-2 Protocol V2 results

The raw bundles and run reports show held-out sequence-copy accuracy of 43.75%, 50.00% and 53.91% for seeds 7, 13 and 29, respectively (mean 49.22%). The random-transition control accuracy was 0% in each report.

A Layer 3 MLP candidate showed positive selected-versus-random copy-margin effects in all three seed reports:

| Seed | Additional margin drop vs matched random component | 95% CI | BH q-value |
|---:|---:|---:|---:|
| 7 | 0.2427 | [0.1890, 0.2992] | 0.000320 |
| 13 | 0.1967 | [0.1458, 0.2505] | 0.000200 |
| 29 | 0.3544 | [0.2718, 0.4467] | 0.000229 |

The corresponding copy-specificity effects were 0.2486, 0.2301 and 0.3578, with BH-adjusted q-values below 0.001 in each run. These results make the Layer 3 MLP candidate a promising repeatable component-level finding in GPT-2. The neuron is Layer 3 neuron 1718, selected by discovery in all three seeds (verified in the raw candidate files). Top-1 flips are modest, so this does not establish a complete circuit.

Attention-head effects were less stable across seeds. Layer 6 Head 9 was selected in all three seeds. Its copy-minus-control specificity was +0.15 (q = 0.16, not significant), +0.36 and +0.45 (q = 0.0002) for seeds 7, 13 and 29. In seed 7 the matched random head (L6 H10) happened to have a larger margin drop than the selected head. Treat it as a head that replicates in two of three seeds. Layer 6 Head 9 also has the largest head-level patching recovery (about 0.12-0.14), but this is small relative to whole-block recovery. Layer 11 Head 10 was selected in all three seeds with strongly negative specificity (ablating it hurts the control task more than the copy task), so it is a generic final-layer sensitivity head, not a copy head. Avoid claiming a single universally important GPT-2 attention head.

## SmolLM3-3B-Base Protocol V2 results (raw bundles included)

The three raw SmolLM3 V2 bundles and the consolidated summary report held-out copy accuracy of approximately 0.44, 0.46 and 0.46 across seeds 7, 13 and 29. Mean target probability was approximately 5–7%, with negative mean copy margins. The behaviour is therefore partial and the baseline must remain central to interpretation.

The most consistent component is **Layer 15, Head 11**:

- selected by discovery in all three seeds;
- selected-versus-matched-random copy-margin drop of approximately 0.29–0.31;
- copy-minus-control specificity of approximately +0.26 to +0.32;
- BH-adjusted q-values below 0.001.

This supports the qualified statement that L15 H11 causally contributes to the measured synthetic copying behaviour relative to the specified controls. It does not show that the head alone implements induction. Ablation changed top-1 prediction on only about 6% of prompts, and copy accuracy shifted from about 0.44 to 0.39.

A second head, L5 H4, had a consistently negative effect and was more associated with generic control-task sensitivity than copy-specific behaviour. MLP neuron selections recurred for some coordinates, but causal effects did not replicate consistently; no neuron is claimed as copy-specific.

## Activation-patching interpretation

The SmolLM3 V2 patching results show large early-layer whole-block successor-position recovery (roughly 0.9–1.0 through about layer 20), declining in later layers, with near-zero random-position recovery. Component-specific head/neuron patches recovered little. This pattern is compatible with information being represented in a distributed or redundant way, but the current patch design and sample size (12 prompts per seed) do not establish a specific circuit path. Layer-35 edge cases and unusually high-activation neuron 505 are excluded from copying-circuit claims.

## Limitations

- The synthetic sequence-copying task is narrow and does not establish natural-language induction or broad language understanding.
- GPT-2 V2 runs used CPU/float32 and did not record a resolved model revision, so exact GPT-2 weights are not pinned. Top-1 flip rates for the L3 neuron and L6 H9 ablations are small (about 2-14%).
- SmolLM3 V2 raw per-seed bundles are included and the headline figures were cross-checked against them. Inference is within-seed; no pooled meta-analysis has been run, and activation patching used only 12 prompts per seed.
- The random-transition control has 0% top-1 accuracy and may be at a floor; it is a generic sensitivity control, not a behaviourally matched natural-text benchmark.
- Only one candidate per layer was tested and only eight layers were screened.
- Activation patching uses a small sample and component-specific patches yielded little recovery.
- Ablation can be off-manifold. Negative and non-specific effects must not be interpreted as evidence for a copying circuit.
- BH correction is reported within each seed; a pooled or meta-analytic analysis across seeds would strengthen claims.

## Conclusions

The current evidence supports a modest, qualified conclusion: selected internal components can causally influence this synthetic copying task, with a repeatable GPT-2 Layer 3 MLP effect across all three raw GPT-2 seeds and a repeatably selected SmolLM3 Layer 15 Head 11 across all three raw SmolLM3 seeds. The evidence is not yet sufficient to claim discovery of a complete induction circuit. The next strongest research extensions are wider candidate screening around SmolLM3 layers 12–18, combined ablations, path patching, more patching prompts, and a natural-text induction benchmark with a better-matched control.
