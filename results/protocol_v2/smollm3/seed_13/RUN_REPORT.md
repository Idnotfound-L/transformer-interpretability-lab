# Sequence-copying mechanistic interpretability study — Protocol V2

- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Architecture: `SmolLM3ForCausalLM`
- Selected layers: `[0, 5, 10, 15, 20, 25, 30, 35]`
- Discovery/copy-eval/control cases: `96/128/128`
- Patch cases: `12`
- Held-out copy accuracy: **0.4609**
- Random-transition control top-1 accuracy: **0.0000**

## Held-out baseline uncertainty

| metric             |   n |       mean |       std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|----------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.460938  | 0.50043   |  0.375     |   0.546875  |
| target_probability | 128 |  0.0604068 | 0.0846857 |  0.0463778 |   0.0759634 |
| copy_margin        | 128 | -0.526123  | 2.85741   | -1.06863   |  -0.0803162 |

## Candidate vs random-control ablation effects on copy task

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |        q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|------------:|
| head   |      15 | head_L15_R1   |         128 |                              0.314697    |  0.247308   |   0.386963  |      9.999e-05  | 0.00026664  |
| head   |      30 | head_L30_R1   |         128 |                              0.0617676   |  0.0380798  |   0.0852051 |      9.999e-05  | 0.00026664  |
| neuron |      20 | neuron_L20_R1 |         128 |                              0.0593262   |  0.036377   |   0.0839844 |      9.999e-05  | 0.00026664  |
| neuron |       5 | neuron_L5_R1  |         128 |                              0.0561523   |  0.0292969  |   0.0839844 |      0.00019998 | 0.000457097 |
| head   |      20 | head_L20_R1   |         128 |                              0.0507812   |  0.0107422  |   0.0908264 |      0.0120988  | 0.021509    |
| neuron |      25 | neuron_L25_R1 |         128 |                              0.0292969   |  0.0109863  |   0.046875  |      0.0019998  | 0.0039996   |
| neuron |       0 | neuron_L0_R1  |         128 |                              0.0175781   | -0.0107422  |   0.0454102 |      0.240876   | 0.256934    |
| neuron |      15 | neuron_L15_R1 |         128 |                              0.015625    | -0.00708618 |   0.0371094 |      0.167483   | 0.219055    |
| neuron |      10 | neuron_L10_R1 |         128 |                              0.0153809   | -0.00610352 |   0.0375977 |      0.177982   | 0.219055    |
| head   |      25 | head_L25_R1   |         128 |                             -0.000244141 | -0.0192871  |   0.0195312 |      1          | 1           |
| neuron |      30 | neuron_L30_R1 |         128 |                             -0.0197754   | -0.0390625  |  -0.0012146 |      0.0537946  | 0.0782467   |
| head   |      10 | head_L10_R1   |         128 |                             -0.0241699   | -0.0615234  |   0.012207  |      0.208879   | 0.238719    |

## Copy-task specificity vs random-transition control

| kind   |   layer | pair_id       |   n_copy_prompts |   n_control_prompts |   copy_minus_control_specificity |    ci95_low |   ci95_high |   permutation_p |   copy_selected_vs_random_mean |   control_selected_vs_random_mean |        q_bh |
|:-------|--------:|:--------------|-----------------:|--------------------:|---------------------------------:|------------:|------------:|----------------:|-------------------------------:|----------------------------------:|------------:|
| head   |      15 | head_L15_R1   |              128 |                 128 |                      0.278648    |  0.19887    |  0.36528    |      9.999e-05  |                    0.314697    |                        0.0360489  | 0.000319968 |
| head   |      30 | head_L30_R1   |              128 |                 128 |                      0.0769348   |  0.0439896  |  0.107773   |      9.999e-05  |                    0.0617676   |                       -0.0151672  | 0.000319968 |
| neuron |       5 | neuron_L5_R1  |              128 |                 128 |                      0.0529556   |  0.0216265  |  0.0854881  |      0.00119988 |                    0.0561523   |                        0.00319672 | 0.00274258  |
| neuron |      20 | neuron_L20_R1 |              128 |                 128 |                      0.0521317   |  0.0245714  |  0.0802776  |      0.00039996 |                    0.0593262   |                        0.00719452 | 0.00106656  |
| neuron |      15 | neuron_L15_R1 |              128 |                 128 |                      0.0263138   | -0.00835514 |  0.0618368  |      0.137186   |                    0.015625    |                       -0.0106888  | 0.219498    |
| neuron |      25 | neuron_L25_R1 |              128 |                 128 |                      0.0207748   | -0.00311432 |  0.0440067  |      0.0931907  |                    0.0292969   |                        0.00852203 | 0.165672    |
| head   |      20 | head_L20_R1   |              128 |                 128 |                      0.0202484   | -0.028978   |  0.068544   |      0.414959   |                    0.0507812   |                        0.0305328  | 0.510718    |
| neuron |      10 | neuron_L10_R1 |              128 |                 128 |                      0.0160141   | -0.0148039  |  0.0467169  |      0.29717    |                    0.0153809   |                       -0.00063324 | 0.396227    |
| head   |      10 | head_L10_R1   |              128 |                 128 |                      0.0134754   | -0.030883   |  0.0583244  |      0.549745   |                   -0.0241699   |                       -0.0376453  | 0.62828     |
| head   |      25 | head_L25_R1   |              128 |                 128 |                      0.00592804  | -0.0195938  |  0.0318455  |      0.645735   |                   -0.000244141 |                       -0.00617218 | 0.688784    |
| neuron |       0 | neuron_L0_R1  |              128 |                 128 |                     -0.000963211 | -0.0377088  |  0.0361215  |      0.957104   |                    0.0175781   |                        0.0185413  | 0.957104    |
| neuron |      30 | neuron_L30_R1 |              128 |                 128 |                     -0.0247407   | -0.0511482  |  0.00199219 |      0.069793   |                   -0.0197754   |                        0.00496531 | 0.139586    |

## Activation patching

|   layer | patch_location           | kind   | selection      |   component |   n |   mean_margin_recovery |   margin_recovery_ci95_low |   margin_recovery_ci95_high |   mean_target_logit_recovery |   target_logit_recovery_ci95_low |   target_logit_recovery_ci95_high |
|--------:|:-------------------------|:-------|:---------------|------------:|----:|-----------------------:|---------------------------:|----------------------------:|-----------------------------:|---------------------------------:|----------------------------------:|
|       0 | block_random_position    | block  | clean_patch    |         nan |  12 |            -0.00980074 |               -0.0828277   |                  0.0493712  |                 -0.0425965   |                     -0.134583    |                       0.00757348  |
|       0 | block_successor_position | block  | clean_patch    |         nan |  12 |             1.002      |                0.975263    |                  1.0399     |                  0.96994     |                      0.946259    |                       0.991619    |
|       0 | head_context_final_query | head   | discovery_top  |           1 |  12 |             0.00870536 |                0.000821499 |                  0.0193797  |                 -0.00433553  |                     -0.0149931   |                       0.00482753  |
|       0 | head_context_final_query | head   | random_control |           9 |  12 |             0.0233802  |                0.00413982  |                  0.0578408  |                 -0.00229895  |                     -0.0168537   |                       0.00946628  |
|       0 | mlp_neuron_final_query   | neuron | discovery_top  |        9146 |  12 |             0.00478437 |               -0.00270333  |                  0.0129625  |                 -0.00627389  |                     -0.0222382   |                       0.00522918  |
|       0 | mlp_neuron_final_query   | neuron | random_control |       10379 |  12 |             0.00957166 |                0.00064913  |                  0.0247043  |                  0.000988279 |                     -0.00505675  |                       0.00637826  |
|       5 | block_random_position    | block  | clean_patch    |         nan |  12 |             0.0209759  |               -0.000179335 |                  0.0535382  |                  0.00312304  |                     -0.0078411   |                       0.0126439   |
|       5 | block_successor_position | block  | clean_patch    |         nan |  12 |             0.863939   |                0.58029     |                  1.02512    |                  1.0813      |                      0.974478    |                       1.20978     |
|       5 | head_context_final_query | head   | discovery_top  |           4 |  12 |             0.0142707  |               -0.00585506  |                  0.0469653  |                 -0.0105259   |                     -0.0229272   |                      -0.00119358  |
|       5 | head_context_final_query | head   | random_control |          13 |  12 |             0.00199696 |               -0.00162638  |                  0.00561867 |                 -0.00194848  |                     -0.0143575   |                       0.00638154  |
|       5 | mlp_neuron_final_query   | neuron | discovery_top  |        6086 |  12 |             0.0128331  |               -0.00173133  |                  0.0346304  |                 -0.00896477  |                     -0.0272351   |                       0.00481888  |
|       5 | mlp_neuron_final_query   | neuron | random_control |        1892 |  12 |             0.00938346 |               -0.00168107  |                  0.0261431  |                 -0.00864347  |                     -0.0240093   |                       0.00331815  |
|      10 | block_random_position    | block  | clean_patch    |         nan |  12 |             0.0224956  |               -0.00421331  |                  0.0575668  |                  0.00572605  |                     -0.0154887   |                       0.0290918   |
|      10 | block_successor_position | block  | clean_patch    |         nan |  12 |             0.818427   |                0.445889    |                  1.04639    |                  1.13128     |                      0.97904     |                       1.3087      |
|      10 | head_context_final_query | head   | discovery_top  |           1 |  12 |             0.00620098 |               -0.00193562  |                  0.017498   |                 -0.00841029  |                     -0.0212277   |                       0.00185953  |
|      10 | head_context_final_query | head   | random_control |          12 |  12 |             0.0131623  |                0.000540159 |                  0.0337844  |                 -0.00245649  |                     -0.015188    |                       0.00677761  |
|      10 | mlp_neuron_final_query   | neuron | discovery_top  |        6073 |  12 |             0.0101915  |               -0.000422072 |                  0.0264858  |                 -0.00797631  |                     -0.0233606   |                       0.00458476  |
|      10 | mlp_neuron_final_query   | neuron | random_control |        5200 |  12 |            -0.00161126 |               -0.0086509   |                  0.00627638 |                 -0.00765062  |                     -0.0306332   |                       0.00632623  |
|      15 | block_random_position    | block  | clean_patch    |         nan |  12 |             0.0209399  |                0.00139923  |                  0.0430663  |                  0.00824591  |                     -0.0154241   |                       0.0355423   |
|      15 | block_successor_position | block  | clean_patch    |         nan |  12 |             0.621018   |               -0.0992585   |                  1.01213    |                  1.19644     |                      0.96303     |                       1.50401     |
|      15 | head_context_final_query | head   | discovery_top  |          11 |  12 |             0.0394509  |               -0.0224871   |                  0.146389   |                 -0.030655    |                     -0.0675087   |                      -0.000615081 |
|      15 | head_context_final_query | head   | random_control |           7 |  12 |             0.00772023 |               -0.000468013 |                  0.0188134  |                  0.00649984  |                      4.80708e-05 |                       0.0142294   |
|      15 | mlp_neuron_final_query   | neuron | discovery_top  |        1447 |  12 |             0.0172926  |               -0.00042042  |                  0.0477834  |                 -0.00787102  |                     -0.020761    |                       0.00217459  |
|      15 | mlp_neuron_final_query   | neuron | random_control |        1550 |  12 |             0.00504342 |               -0.00244538  |                  0.0160247  |                 -0.000240962 |                     -0.0092085   |                       0.007166    |
|      20 | block_random_position    | block  | clean_patch    |         nan |  12 |             0.0106601  |               -0.00561857  |                  0.0288252  |                  0.00262444  |                     -0.0147073   |                       0.0196168   |
|      20 | block_successor_position | block  | clean_patch    |         nan |  12 |             0.650007   |                0.0147805   |                  1.00523    |                  1.15856     |                      0.935271    |                       1.44527     |
|      20 | head_context_final_query | head   | discovery_top  |           1 |  12 |             0.0121367  |                0.000470885 |                  0.0287676  |                 -0.00478348  |                     -0.0188943   |                       0.00619415  |
|      20 | head_context_final_query | head   | random_control |           0 |  12 |             0.00469198 |               -0.00851016  |                  0.0232873  |                 -0.00642384  |                     -0.0179857   |                       0.00197496  |
|      20 | mlp_neuron_final_query   | neuron | discovery_top  |        2209 |  12 |             0.0121847  |               -0.00227846  |                  0.0345377  |                 -0.00115767  |                     -0.0182305   |                       0.0123497   |
|      20 | mlp_neuron_final_query   | neuron | random_control |        6814 |  12 |             0.00627942 |               -0.00206442  |                  0.0180298  |                 -0.00523568  |                     -0.0156079   |                       0.003019    |
|      25 | block_random_position    | block  | clean_patch    |         nan |  12 |             0.00400866 |               -0.00404914  |                  0.0124617  |                  0.00678831  |                     -0.000573431 |                       0.01444     |
|      25 | block_successor_position | block  | clean_patch    |         nan |  12 |             0.227333   |               -0.367494    |                  0.559953   |                  0.512582    |                      0.317259    |                       0.743871    |
|      25 | head_context_final_query | head   | discovery_top  |           8 |  12 |             0.0162051  |                0.000214931 |                  0.0418005  |                  0.0018534   |                     -0.00468607  |                       0.00823135  |
|      25 | head_context_final_query | head   | random_control |           6 |  12 |             0.00768538 |               -0.00357994  |                  0.0244701  |                 -0.0081873   |                     -0.0207442   |                       0.00133924  |
|      25 | mlp_neuron_final_query   | neuron | discovery_top  |        9433 |  12 |             0.0106405  |                0.00271999  |                  0.0211569  |                 -0.00343921  |                     -0.012376    |                       0.00403899  |
|      25 | mlp_neuron_final_query   | neuron | random_control |        2084 |  12 |             0.0070741  |               -0.00226351  |                  0.0192221  |                 -0.000603923 |                     -0.00926835  |                       0.00604983  |
|      30 | block_random_position    | block  | clean_patch    |         nan |  12 |             0.00824404 |                0.00027826  |                  0.0169324  |                  0.00405572  |                     -0.00189835  |                       0.00958053  |
|      30 | block_successor_position | block  | clean_patch    |         nan |  12 |             0.217761   |               -0.0088818   |                  0.36521    |                  0.313525    |                      0.220568    |                       0.406483    |
|      30 | head_context_final_query | head   | discovery_top  |           9 |  12 |             0.00804521 |                0.00253121  |                  0.0134497  |                  0.00346916  |                     -0.0105572   |                       0.0131664   |
|      30 | head_context_final_query | head   | random_control |           4 |  12 |             0.0142016  |                0.00540176  |                  0.0252189  |                  0.00557707  |                     -0.00660069  |                       0.0174949   |
|      30 | mlp_neuron_final_query   | neuron | discovery_top  |        8506 |  12 |             0.0178337  |                0.00218665  |                  0.0442519  |                 -0.00599877  |                     -0.0246378   |                       0.0059074   |
|      30 | mlp_neuron_final_query   | neuron | random_control |        1137 |  12 |             0.0124359  |               -0.00284857  |                  0.0387902  |                 -0.00961271  |                     -0.0276076   |                       0.00224527  |
|      35 | block_random_position    | block  | clean_patch    |         nan |  12 |             0          |                0           |                  0          |                  0           |                      0           |                       0           |
|      35 | block_successor_position | block  | clean_patch    |         nan |  12 |             0          |                0           |                  0          |                  0           |                      0           |                       0           |
|      35 | head_context_final_query | head   | discovery_top  |           0 |  12 |            -0.0870874  |               -0.14024     |                 -0.0255731  |                 -0.138708    |                     -0.228168    |                      -0.0763089   |
|      35 | head_context_final_query | head   | random_control |           2 |  12 |             0.00931453 |               -0.00635099  |                  0.0238865  |                  0.0115635   |                     -0.00181255  |                       0.0242811   |
|      35 | mlp_neuron_final_query   | neuron | discovery_top  |         505 |  12 |            -0.010362   |               -0.0185566   |                 -0.00218009 |                 -0.0424829   |                     -0.109056    |                      -0.000516482 |
|      35 | mlp_neuron_final_query   | neuron | random_control |        9554 |  12 |             0.0052485  |               -0.00658679  |                  0.02226    |                 -0.0194119   |                     -0.0437067   |                      -0.00165282  |

## Interpretation guidance

A component is more specifically implicated in sequence copying when its selected-vs-control ablation effect is larger on copy prompts than on independent random-transition prompts, with uncertainty intervals and corrected tests supporting the difference.
The successor-B attention score is a candidate-discovery signal only; it is not proof of an induction circuit.
Component-specific patching and non-trivial position controls provide additional causal evidence, but should be interpreted alongside baseline performance and held-out ablations.
