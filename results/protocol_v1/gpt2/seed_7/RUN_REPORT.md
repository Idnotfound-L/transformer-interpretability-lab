# Sequence-copying mechanistic interpretability study

- Model: `gpt2`
- Architecture: `GPT2LMHeadModel`
- Layers screened: `[0, 2, 3, 5, 6, 8, 9, 11]`
- Discovery cases: `96`
- Held-out evaluation cases: `128`
- Patch cases: `12`
- Baseline held-out copy accuracy: **0.6172**

## Held-out baseline uncertainty

| metric             |   n |      mean |       std |   ci95_low |   ci95_high |
|:-------------------|----:|----------:|----------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.617188 | 0.487983  |   0.53125  |   0.703125  |
| target_probability | 128 |  0.07446  | 0.0883543 |   0.059895 |   0.0900084 |
| copy_margin        | 128 | -0.228004 | 2.21369   |  -0.621906 |   0.152813  |

## Paired candidate-vs-random-control results

| kind   |   layer | pair_id      |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |       q_bh |
|:-------|--------:|:-------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|-----------:|
| head   |      11 | head_L11_R1  |         128 |                                0.492529  |  0.404077   |   0.579479  |       9.999e-05 | 0.00017776 |
| neuron |       3 | neuron_L3_R1 |         128 |                                0.310002  |  0.226406   |   0.402909  |       9.999e-05 | 0.00017776 |
| head   |       2 | head_L2_R1   |         128 |                                0.298171  | -0.0706605  |   0.680786  |       0.141086  | 0.173644   |
| neuron |       8 | neuron_L8_R1 |         128 |                                0.192282  |  0.130942   |   0.254144  |       9.999e-05 | 0.00017776 |
| neuron |       0 | neuron_L0_R1 |         128 |                                0.0685847 |  0.0403808  |   0.0999125 |       9.999e-05 | 0.00017776 |
| neuron |       2 | neuron_L2_R1 |         128 |                                0.0569453 |  0.0322035  |   0.0852313 |       9.999e-05 | 0.00017776 |
| head   |       3 | head_L3_R1   |         128 |                                0.0436044 | -0.00351136 |   0.0905859 |       0.0693931 | 0.100935   |
| head   |       5 | head_L5_R1   |         128 |                                0.0406893 | -0.0469309  |   0.12844   |       0.381762  | 0.407213   |
| head   |       9 | head_L9_R1   |         128 |                                0.0198908 | -0.00240433 |   0.0430809 |       0.0915908 | 0.122121   |
| neuron |       9 | neuron_L9_R1 |         128 |                               -0.0220447 | -0.0571127  |   0.0100011 |       0.206079  | 0.235519   |
| head   |       8 | head_L8_R1   |         128 |                               -0.0285444 | -0.0945285  |   0.0356202 |       0.407559  | 0.407559   |
| head   |       6 | head_L6_R1   |         128 |                               -0.111312  | -0.196596   |  -0.0264756 |       0.0140986 | 0.0225577  |

## Interpretation guidance

A positive selected-minus-random margin-drop estimate means ablating the discovery-ranked component reduced the correct-token margin more than ablating its same-layer random control. Inspect the confidence interval and BH-adjusted q-value before interpreting the result. A negative result is still informative and should be reported honestly.

Attention-to-first-A is a descriptive ranking signal, not causal proof. Causal evidence comes from held-out final-query ablations and clean-to-corrupted activation patching.
