# Sequence-copying mechanistic interpretability study

- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Architecture: `SmolLM3ForCausalLM`
- Layers screened: `[0, 5, 10, 15, 20, 25, 30, 35]`
- Discovery cases: `96`
- Held-out evaluation cases: `128`
- Patch cases: `12`
- Baseline held-out copy accuracy: **0.3984**

## Held-out baseline uncertainty

| metric             |   n |       mean |       std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|----------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.398438  | 0.4915    |  0.3125    |   0.484375  |
| target_probability | 128 |  0.0479814 | 0.0658819 |  0.0373978 |   0.0601148 |
| copy_margin        | 128 | -1.04932   | 2.95709   | -1.58032   |  -0.557605  |

## Paired candidate-vs-random-control results

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |        q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|------------:|
| head   |       5 | head_L5_R1    |         128 |                              0.653931    |  0.523175   |  0.791629   |       9.999e-05 | 0.000319968 |
| neuron |      20 | neuron_L20_R1 |         128 |                              0.0606689   |  0.0410156  |  0.0808136  |       9.999e-05 | 0.000319968 |
| neuron |      10 | neuron_L10_R1 |         128 |                              0.0252686   |  0.00451355 |  0.0454132  |       0.0172983 | 0.0438813   |
| neuron |       5 | neuron_L5_R1  |         128 |                              0.0214844   | -0.0180664  |  0.0678711  |       0.388061  | 0.517415    |
| neuron |      30 | neuron_L30_R1 |         128 |                              0.00964355  | -0.0202637  |  0.0408966  |       0.538646  | 0.629537    |
| neuron |      25 | neuron_L25_R1 |         128 |                              0.00500488  | -0.0111084  |  0.0200226  |       0.550845  | 0.629537    |
| head   |      30 | head_L30_R1   |         128 |                              0.000488281 | -0.0234436  |  0.0244141  |       0.984902  | 0.984902    |
| head   |      20 | head_L20_R1   |         128 |                             -0.00598145  | -0.0272217  |  0.0157471  |       0.59834   | 0.63823     |
| neuron |       0 | neuron_L0_R1  |         128 |                             -0.0275879   | -0.0612793  |  0.00317383 |       0.10159   | 0.180604    |
| head   |      35 | head_L35_R1   |         128 |                             -0.02771     | -0.0671387  |  0.0065918  |       0.153685  | 0.23474     |
| neuron |      15 | neuron_L15_R1 |         128 |                             -0.0319824   | -0.0598145  | -0.00463867 |       0.0191981 | 0.0438813   |
| head   |      10 | head_L10_R1   |         128 |                             -0.0324707   | -0.0629883  | -0.00268555 |       0.0329967 | 0.0659934   |

## Interpretation guidance

A positive selected-minus-random margin-drop estimate means ablating the discovery-ranked component reduced the correct-token margin more than ablating its same-layer random control. Inspect the confidence interval and BH-adjusted q-value before interpreting the result. A negative result is still informative and should be reported honestly.

Attention-to-first-A is a descriptive ranking signal, not causal proof. Causal evidence comes from held-out final-query ablations and clean-to-corrupted activation patching.
