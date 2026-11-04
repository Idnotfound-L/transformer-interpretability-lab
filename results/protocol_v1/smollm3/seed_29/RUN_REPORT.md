# Sequence-copying mechanistic interpretability study

- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Architecture: `SmolLM3ForCausalLM`
- Layers screened: `[0, 5, 10, 15, 20, 25, 30, 35]`
- Discovery cases: `96`
- Held-out evaluation cases: `128`
- Patch cases: `12`
- Baseline held-out copy accuracy: **0.4062**

## Held-out baseline uncertainty

| metric             |   n |       mean |      std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|---------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.40625   | 0.493062 |  0.320312  |   0.492188  |
| target_probability | 128 |  0.0643815 | 0.117456 |  0.0457944 |   0.0855855 |
| copy_margin        | 128 | -0.904541  | 2.57206  | -1.365     |  -0.459454  |

## Paired candidate-vs-random-control results

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |   ci95_low |    ci95_high |   permutation_p |        q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|-----------:|-------------:|----------------:|------------:|
| head   |       5 | head_L5_R1    |         128 |                              0.444092    |  0.317865  |  0.574463    |      9.999e-05  | 0.000319968 |
| neuron |      20 | neuron_L20_R1 |         128 |                              0.072998    |  0.0473633 |  0.0979004   |      9.999e-05  | 0.000319968 |
| neuron |       5 | neuron_L5_R1  |         128 |                              0.0393066   |  0.0024353 |  0.0820312   |      0.0532947  | 0.0878458   |
| neuron |      10 | neuron_L10_R1 |         128 |                              0.027832    |  0.0114746 |  0.0444336   |      0.00129987 | 0.00346632  |
| neuron |      30 | neuron_L30_R1 |         128 |                              0.0158691   | -0.015625  |  0.046875    |      0.318868   | 0.392453    |
| head   |      30 | head_L30_R1   |         128 |                              0.000732422 | -0.020752  |  0.0222229   |      0.968503   | 0.968503    |
| head   |      35 | head_L35_R1   |         128 |                             -0.00292969  | -0.0263672 |  0.0200195   |      0.836716   | 0.892497    |
| head   |      25 | head_L25_R1   |         128 |                             -0.0134277   | -0.0349182 |  0.00732422  |      0.229077   | 0.305436    |
| neuron |      25 | neuron_L25_R1 |         128 |                             -0.0163574   | -0.0324707 |  0.000250244 |      0.0584942  | 0.0878458   |
| neuron |       0 | neuron_L0_R1  |         128 |                             -0.0283203   | -0.0546875 | -0.00195312  |      0.0414959  | 0.0829917   |
| neuron |      15 | neuron_L15_R1 |         128 |                             -0.0290527   | -0.0534729 | -0.00683594  |      0.0129987  | 0.0297113   |
| head   |      10 | head_L10_R1   |         128 |                             -0.0307617   | -0.0620117 |  0.0012207   |      0.060394   | 0.0878458   |

## Interpretation guidance

A positive selected-minus-random margin-drop estimate means ablating the discovery-ranked component reduced the correct-token margin more than ablating its same-layer random control. Inspect the confidence interval and BH-adjusted q-value before interpreting the result. A negative result is still informative and should be reported honestly.

Attention-to-first-A is a descriptive ranking signal, not causal proof. Causal evidence comes from held-out final-query ablations and clean-to-corrupted activation patching.
