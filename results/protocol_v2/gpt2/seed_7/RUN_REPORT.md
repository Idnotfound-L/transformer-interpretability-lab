# Sequence-copying mechanistic interpretability study — Protocol V2

- Model: `gpt2`
- Architecture: `GPT2LMHeadModel`
- Selected layers: `[0, 2, 3, 5, 6, 8, 9, 11]`
- Discovery/copy-eval/control cases: `96/128/128`
- Patch cases: `12`
- Held-out copy accuracy: **0.4375**
- Random-transition control top-1 accuracy: **0.0000**

## Held-out baseline uncertainty

| metric             |   n |       mean |      std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|---------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.4375    | 0.498028 |   0.351562 |    0.515625 |
| target_probability | 128 |  0.0888204 | 0.147415 |   0.06512  |    0.115354 |
| copy_margin        | 128 | -0.61964   | 2.54725  |  -1.07828  |   -0.190395 |

## Candidate vs random-control ablation effects on copy task

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |        q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|------------:|
| neuron |       3 | neuron_L3_R1  |         128 |                               0.242704   |  0.188966   |  0.299152   |      9.999e-05  | 0.000319968 |
| head   |       9 | head_L9_R1    |         128 |                               0.0416372  | -0.00415639 |  0.0864926  |      0.079892   | 0.106523    |
| neuron |       8 | neuron_L8_R1  |         128 |                               0.0248528  |  0.0055886  |  0.0452701  |      0.0157984  | 0.0280861   |
| neuron |       2 | neuron_L2_R1  |         128 |                               0.0143538  | -0.00613906 |  0.0343334  |      0.178482   | 0.21967     |
| neuron |       0 | neuron_L0_R1  |         128 |                               0.00863284 | -0.0166007  |  0.0362062  |      0.558844   | 0.5961      |
| neuron |       5 | neuron_L5_R1  |         128 |                              -0.00204849 | -0.0200988  |  0.0181124  |      0.841616   | 0.841616    |
| neuron |       6 | neuron_L6_R1  |         128 |                              -0.038448   | -0.0612066  | -0.0165395  |      0.00109989 | 0.00219978  |
| head   |       3 | head_L3_R1    |         128 |                              -0.0531432  | -0.108315   | -0.00213725 |      0.0546945  | 0.0795557   |
| head   |       2 | head_L2_R1    |         128 |                              -0.0632566  | -0.103986   | -0.0262497  |      0.00109989 | 0.00219978  |
| head   |       6 | head_L6_R1    |         128 |                              -0.0840435  | -0.245642   |  0.0917394  |      0.362164   | 0.413901    |
| neuron |      11 | neuron_L11_R1 |         128 |                              -0.0950778  | -0.129226   | -0.0645519  |      9.999e-05  | 0.000319968 |
| head   |       8 | head_L8_R1    |         128 |                              -0.0992602  | -0.15989    | -0.0426169  |      0.00069993 | 0.00186648  |

## Copy-task specificity vs random-transition control

| kind   |   layer | pair_id      |   n_copy_prompts |   n_control_prompts |   copy_minus_control_specificity |   ci95_low |   ci95_high |   permutation_p |   copy_selected_vs_random_mean |   control_selected_vs_random_mean |       q_bh |
|:-------|--------:|:-------------|-----------------:|--------------------:|---------------------------------:|-----------:|------------:|----------------:|-------------------------------:|----------------------------------:|-----------:|
| neuron |       3 | neuron_L3_R1 |              128 |                 128 |                       0.248619   |  0.194474  |  0.307186   |      9.999e-05  |                     0.242704   |                       -0.0059149  | 0.00053328 |
| head   |       6 | head_L6_R1   |              128 |                 128 |                       0.153445   | -0.0157301 |  0.345251   |      0.0833917  |                    -0.0840435  |                       -0.237489   | 0.159984   |
| head   |       9 | head_L9_R1   |              128 |                 128 |                       0.0833248  |  0.033317  |  0.134178   |      0.00109989 |                     0.0416372  |                       -0.0416876  | 0.00383962 |
| neuron |       8 | neuron_L8_R1 |              128 |                 128 |                       0.0416472  |  0.015904  |  0.0680499  |      0.00169983 |                     0.0248528  |                       -0.0167945  | 0.00453288 |
| neuron |       9 | neuron_L9_R1 |              128 |                 128 |                       0.0254143  | -0.0236785 |  0.0753541  |      0.310269   |                    -0.10201    |                       -0.127425   | 0.49643    |
| neuron |       2 | neuron_L2_R1 |              128 |                 128 |                       0.00667384 | -0.0175383 |  0.0303995  |      0.59694    |                     0.0143538  |                        0.00767997 | 0.811386   |
| neuron |       0 | neuron_L0_R1 |              128 |                 128 |                       0.00542757 | -0.0226947 |  0.0355729  |      0.729127   |                     0.00863284 |                        0.00320527 | 0.833288   |
| neuron |       5 | neuron_L5_R1 |              128 |                 128 |                       0.00487983 | -0.0164244 |  0.0277937  |      0.684332   |                    -0.00204849 |                       -0.00692832 | 0.833288   |
| head   |       5 | head_L5_R1   |              128 |                 128 |                      -0.00735468 | -0.139948  |  0.112532   |      0.910409   |                    -0.111366   |                       -0.104012   | 0.910409   |
| head   |       0 | head_L0_R1   |              128 |                 128 |                      -0.0133404  | -0.202214  |  0.180362   |      0.89521    |                    -0.216576   |                       -0.203235   | 0.910409   |
| head   |       3 | head_L3_R1   |              128 |                 128 |                      -0.0190412  | -0.0962214 |  0.0517248  |      0.608539   |                    -0.0531432  |                       -0.0341021  | 0.811386   |
| head   |       2 | head_L2_R1   |              128 |                 128 |                      -0.0433003  | -0.0941829 |  0.00519365 |      0.089991   |                    -0.0632566  |                       -0.0199564  | 0.159984   |

## Activation patching

|   layer | patch_location           | kind   | selection      |   component |   n |   mean_margin_recovery |   margin_recovery_ci95_low |   margin_recovery_ci95_high |   mean_target_logit_recovery |   target_logit_recovery_ci95_low |   target_logit_recovery_ci95_high |
|--------:|:-------------------------|:-------|:---------------|------------:|----:|-----------------------:|---------------------------:|----------------------------:|-----------------------------:|---------------------------------:|----------------------------------:|
|       0 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00369268  |                0.00125871  |                 0.00645475  |                 -0.0277533   |                     -0.066984    |                       0.00246558  |
|       0 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.990098    |                0.972841    |                 1.00675     |                  0.812935    |                      0.415979    |                       1.03923     |
|       0 | head_context_final_query | head   | discovery_top  |           3 |  12 |            1.18607e-05 |               -1.46836e-05 |                 4.32526e-05 |                 -7.71475e-05 |                     -0.000305069 |                       7.41953e-05 |
|       0 | head_context_final_query | head   | random_control |           9 |  12 |            0.000473374 |               -0.00452278  |                 0.00469237  |                 -0.0136279   |                     -0.0619701   |                       0.03063     |
|       0 | mlp_neuron_final_query   | neuron | discovery_top  |        1046 |  12 |           -2.59927e-05 |               -7.81324e-05 |                 2.65901e-05 |                 -0.00397116  |                     -0.0127628   |                       0.00077183  |
|       0 | mlp_neuron_final_query   | neuron | random_control |        1994 |  12 |           -1.35202e-05 |               -6.5806e-05  |                 3.45468e-05 |                 -0.000273315 |                     -0.00127209  |                       0.000400051 |
|       2 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00968525  |                0.00104492  |                 0.0213178   |                  0.0770712   |                     -0.108658    |                       0.322196    |
|       2 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.948362    |                0.889452    |                 0.987405    |                  1.83941     |                      0.772004    |                       3.78634     |
|       2 | head_context_final_query | head   | discovery_top  |           7 |  12 |           -0.000912102 |               -0.00177243  |                -0.00013178  |                  0.00873932  |                     -0.00716442  |                       0.033642    |
|       2 | head_context_final_query | head   | random_control |           2 |  12 |            0.00020165  |               -0.000443231 |                 0.000906298 |                  0.0121232   |                     -0.00690814  |                       0.0375434   |
|       2 | mlp_neuron_final_query   | neuron | discovery_top  |        1984 |  12 |            0.000224332 |               -4.58699e-05 |                 0.000516058 |                 -0.0388553   |                     -0.125514    |                       0.0083679   |
|       2 | mlp_neuron_final_query   | neuron | random_control |        1971 |  12 |           -0.00011572  |               -0.00163708  |                 0.00134396  |                  0.00856142  |                      0.00252894  |                       0.0153139   |
|       3 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00788727  |                5.7222e-05  |                 0.0161061   |                  0.065856    |                     -0.0901136   |                       0.240967    |
|       3 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.934764    |                0.886197    |                 0.966069    |                  2.33722     |                      0.690825    |                       5.44104     |
|       3 | head_context_final_query | head   | discovery_top  |          10 |  12 |            0.000573785 |               -0.00227789  |                 0.00349806  |                  0.0219706   |                     -0.00965986  |                       0.0711158   |
|       3 | head_context_final_query | head   | random_control |           8 |  12 |            0.000879047 |               -0.000287052 |                 0.00228669  |                 -0.0154857   |                     -0.0650016   |                       0.0169285   |
|       3 | mlp_neuron_final_query   | neuron | discovery_top  |        1718 |  12 |           -0.000490457 |               -0.00197483  |                 0.000931506 |                  0.0283343   |                      0.00105662  |                       0.0786349   |
|       3 | mlp_neuron_final_query   | neuron | random_control |        1287 |  12 |            0.000337388 |               -7.95889e-05 |                 0.000942904 |                  0.0445678   |                     -0.00139295  |                       0.13452     |
|       5 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00021139  |               -0.0123567   |                 0.0105823   |                  0.023946    |                     -0.123722    |                       0.168809    |
|       5 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.892172    |                0.832398    |                 0.947423    |                  3.15311     |                      0.154986    |                       8.65231     |
|       5 | head_context_final_query | head   | discovery_top  |           5 |  12 |            0.0350979   |                0.0191093   |                 0.0527772   |                  0.500142    |                     -0.0732016   |                       1.53464     |
|       5 | head_context_final_query | head   | random_control |           6 |  12 |           -7.54305e-05 |               -0.000992292 |                 0.00070001  |                 -0.0411233   |                     -0.13628     |                       0.0114877   |
|       5 | mlp_neuron_final_query   | neuron | discovery_top  |         589 |  12 |           -0.000575378 |               -0.00147344  |                 0.000197548 |                 -0.0269695   |                     -0.101375    |                       0.0213449   |
|       5 | mlp_neuron_final_query   | neuron | random_control |        1016 |  12 |            9.18028e-05 |               -0.000409631 |                 0.000549138 |                  0.000572274 |                     -0.00603111  |                       0.00788942  |
|       6 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.000370052 |               -0.01109     |                 0.0100892   |                  0.0115613   |                     -0.144047    |                       0.17578     |
|       6 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.72754     |                0.618042    |                 0.826259    |                  2.6904      |                     -0.107829    |                       7.88262     |
|       6 | head_context_final_query | head   | discovery_top  |           9 |  12 |            0.124208    |                0.0904628   |                 0.163216    |                  1.35964     |                     -0.0720636   |                       4.00416     |
|       6 | head_context_final_query | head   | random_control |          10 |  12 |            0.0141499   |                0.00364538  |                 0.0258815   |                  0.341899    |                     -0.0492043   |                       1.032       |
|       6 | mlp_neuron_final_query   | neuron | discovery_top  |        2527 |  12 |           -0.00153437  |               -0.00530316  |                 0.00217497  |                 -0.191196    |                     -0.576033    |                       0.0365676   |
|       6 | mlp_neuron_final_query   | neuron | random_control |        1462 |  12 |            0.000972943 |               -0.000632451 |                 0.00284912  |                  0.0991046   |                      0.000971601 |                       0.289872    |
|       8 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00715118  |               -0.000117933 |                 0.0167023   |                  0.110501    |                     -0.0645989   |                       0.37901     |
|       8 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.479085    |                0.366104    |                 0.591007    |                  4.26899     |                     -0.238234    |                      12.8919      |
|       8 | head_context_final_query | head   | discovery_top  |           6 |  12 |            0.0507811   |                0.0308355   |                 0.0708746   |                 -0.0450347   |                     -0.363725    |                       0.202602    |
|       8 | head_context_final_query | head   | random_control |           1 |  12 |            0.0501408   |                0.0300671   |                 0.0730732   |                 -0.321898    |                     -0.919368    |                       8.19854e-05 |
|       8 | mlp_neuron_final_query   | neuron | discovery_top  |        2038 |  12 |            0.00848944  |                0.00252003  |                 0.016176    |                 -0.0757631   |                     -0.240645    |                       0.0156929   |
|       8 | mlp_neuron_final_query   | neuron | random_control |        2108 |  12 |           -0.000209814 |               -0.000667375 |                 0.000205314 |                  1.80027e-05 |                     -0.00445197  |                       0.00320706  |
|       9 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00573357  |               -0.000539209 |                 0.0129975   |                  0.153543    |                     -0.0289558   |                       0.452439    |
|       9 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.382371    |                0.283705    |                 0.4911      |                  0.432006    |                     -0.260221    |                       1.37985     |
|       9 | head_context_final_query | head   | discovery_top  |           6 |  12 |            0.0680506   |                0.0467745   |                 0.0905254   |                 -0.281697    |                     -0.782599    |                       0.00372924  |
|       9 | head_context_final_query | head   | random_control |           4 |  12 |           -0.000727574 |               -0.00825519  |                 0.00649035  |                 -0.141656    |                     -0.375682    |                       0.00414517  |
|       9 | mlp_neuron_final_query   | neuron | discovery_top  |         840 |  12 |            0.000109679 |               -0.000908155 |                 0.00120344  |                  0.0085843   |                     -0.0142265   |                       0.032758    |
|       9 | mlp_neuron_final_query   | neuron | random_control |        2477 |  12 |            0.000570909 |               -0.00100993  |                 0.00266322  |                 -0.107358    |                     -0.282242    |                       0.00176879  |
|      11 | block_random_position    | block  | clean_patch    |         nan |  12 |            0           |                0           |                 0           |                  0           |                      0           |                       0           |
|      11 | block_successor_position | block  | clean_patch    |         nan |  12 |            0           |                0           |                 0           |                  0           |                      0           |                       0           |
|      11 | head_context_final_query | head   | discovery_top  |          10 |  12 |           -0.0713112   |               -0.108698    |                -0.0379761   |                 -0.156898    |                     -0.407209    |                       0.0171681   |
|      11 | head_context_final_query | head   | random_control |           4 |  12 |           -0.0024182   |               -0.00530445  |                 0.000469385 |                 -0.528698    |                     -1.51882     |                      -0.00161564  |
|      11 | mlp_neuron_final_query   | neuron | discovery_top  |        1998 |  12 |           -0.00617773  |               -0.0110467   |                -0.00174167  |                 -0.0108692   |                     -0.114831    |                       0.0670349   |
|      11 | mlp_neuron_final_query   | neuron | random_control |         965 |  12 |           -0.000176173 |               -0.00140178  |                 0.00109961  |                 -0.017204    |                     -0.0431766   |                      -0.0016741   |

## Interpretation guidance

A component is more specifically implicated in sequence copying when its selected-vs-control ablation effect is larger on copy prompts than on independent random-transition prompts, with uncertainty intervals and corrected tests supporting the difference.
The successor-B attention score is a candidate-discovery signal only; it is not proof of an induction circuit.
Component-specific patching and non-trivial position controls provide additional causal evidence, but should be interpreted alongside baseline performance and held-out ablations.
