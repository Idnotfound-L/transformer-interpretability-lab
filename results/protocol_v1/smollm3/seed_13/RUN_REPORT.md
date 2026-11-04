# Sequence-copying mechanistic interpretability study

- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Architecture: `SmolLM3ForCausalLM`
- Layers screened: `[0, 5, 10, 15, 20, 25, 30, 35]`
- Discovery cases: `96`
- Held-out evaluation cases: `128`
- Patch cases: `12`
- Baseline held-out copy accuracy: **0.4219**

## Held-out baseline uncertainty

| metric             |   n |       mean |      std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|---------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.421875  | 0.495799 |  0.335938  |    0.507812 |
| target_probability | 128 |  0.0688416 | 0.13409  |  0.0468989 |    0.093405 |
| copy_margin        | 128 | -0.941406  | 2.81956  | -1.43655   |   -0.469727 |

## Paired candidate-vs-random-control results

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |       q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|-----------:|
| head   |       5 | head_L5_R1    |         128 |                               0.87207    |  0.66748    |  1.1089     |      9.999e-05  | 0.00026664 |
| neuron |      20 | neuron_L20_R1 |         128 |                               0.0623779  |  0.0395477  |  0.0854523  |      9.999e-05  | 0.00026664 |
| neuron |       5 | neuron_L5_R1  |         128 |                               0.0499268  |  0.0217285  |  0.0797119  |      0.00129987 | 0.00231088 |
| head   |      20 | head_L20_R1   |         128 |                               0.0483398  |  0.00927734 |  0.0864258  |      0.0167983  | 0.0223978  |
| neuron |      10 | neuron_L10_R1 |         128 |                               0.0429688  |  0.0258789  |  0.0595703  |      9.999e-05  | 0.00026664 |
| head   |      30 | head_L30_R1   |         128 |                               0.0373535  |  0.0146484  |  0.0617676  |      0.00179982 | 0.00287971 |
| neuron |      30 | neuron_L30_R1 |         128 |                               0.02771    | -0.0045166  |  0.0595764  |      0.0920908  | 0.105247   |
| head   |      25 | head_L25_R1   |         128 |                               0.0109863  | -0.00878906 |  0.0302734  |      0.292471   | 0.311969   |
| neuron |      25 | neuron_L25_R1 |         128 |                              -0.00427246 | -0.0209961  |  0.0124512  |      0.615738   | 0.615738   |
| neuron |      15 | neuron_L15_R1 |         128 |                              -0.038208   | -0.0612793  | -0.0161133  |      0.00129987 | 0.00231088 |
| neuron |       0 | neuron_L0_R1  |         128 |                              -0.0402832  | -0.0778809  | -0.00683594 |      0.0209979  | 0.0258436  |
| head   |      10 | head_L10_R1   |         128 |                              -0.0526123  | -0.0930176  | -0.0126923  |      0.0131987  | 0.0191981  |

## Interpretation guidance

A positive selected-minus-random margin-drop estimate means ablating the discovery-ranked component reduced the correct-token margin more than ablating its same-layer random control. Inspect the confidence interval and BH-adjusted q-value before interpreting the result. A negative result is still informative and should be reported honestly.

Attention-to-first-A is a descriptive ranking signal, not causal proof. Causal evidence comes from held-out final-query ablations and clean-to-corrupted activation patching.
