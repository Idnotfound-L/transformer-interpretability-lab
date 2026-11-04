# Sequence-copying mechanistic interpretability study

- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Architecture: `SmolLM3ForCausalLM`
- Layers screened: `[0, 35]`
- Discovery cases: `16`
- Held-out evaluation cases: `24`
- Patch cases: `2`
- Baseline held-out copy accuracy: **0.4583**

## Held-out baseline uncertainty

| metric             |   n |       mean |       std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|----------:|-----------:|------------:|
| copy_accuracy      |  24 |  0.458333  | 0.508977  |  0.25      |   0.666667  |
| target_probability |  24 |  0.0526281 | 0.0685932 |  0.0278293 |   0.0819327 |
| copy_margin        |  24 | -0.463542  | 2.07567   | -1.2735    |   0.325521  |

## Paired candidate-vs-random-control results

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |   ci95_low |   ci95_high |   permutation_p |      q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|-----------:|------------:|----------------:|----------:|
| head   |       0 | head_L0_R1    |          24 |                                0.0130208 | -0.0286458 |   0.0546875 |      0.641336   | 0.763024  |
| head   |      35 | head_L35_R1   |          24 |                               -0.0130208 | -0.0755859 |   0.0520833 |      0.763024   | 0.763024  |
| neuron |       0 | neuron_L0_R1  |          24 |                               -0.119792  | -0.21875   |  -0.03125   |      0.0260974  | 0.0521948 |
| neuron |      35 | neuron_L35_R1 |          24 |                               -0.268229  | -0.372396  |  -0.153646  |      0.00049995 | 0.0019998 |

## Interpretation guidance

A positive selected-minus-random margin-drop estimate means ablating the discovery-ranked component reduced the correct-token margin more than ablating its same-layer random control. Inspect the confidence interval and BH-adjusted q-value before interpreting the result. A negative result is still informative and should be reported honestly.

Attention-to-first-A is a descriptive ranking signal, not causal proof. Causal evidence comes from held-out final-query ablations and clean-to-corrupted activation patching.
