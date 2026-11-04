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
| target_probability | 128 |  0.0663777 | 0.0967462 |  0.0507763 |   0.0833432 |
| copy_margin        | 128 | -0.740479  | 2.58869   | -1.19388   |  -0.291986  |

## Candidate vs random-control ablation effects on copy task

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |        q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|------------:|
| head   |      15 | head_L15_R1   |         128 |                               0.290283   |  0.226074   |  0.368896   |       9.999e-05 | 0.000319968 |
| neuron |       0 | neuron_L0_R1  |         128 |                               0.240723   |  0.167969   |  0.319824   |       9.999e-05 | 0.000319968 |
| neuron |      25 | neuron_L25_R1 |         128 |                               0.0234375  |  0.00537109 |  0.0419922  |       0.0187981 | 0.0429671   |
| neuron |       5 | neuron_L5_R1  |         128 |                               0.0119629  | -0.017334   |  0.0490723  |       0.582942  | 0.717467    |
| head   |      30 | head_L30_R1   |         128 |                               0.00537109 | -0.0161133  |  0.0268555  |       0.662134  | 0.723021    |
| neuron |      20 | neuron_L20_R1 |         128 |                               0.00219727 | -0.0166077  |  0.0214844  |       0.842116  | 0.842116    |
| head   |      25 | head_L25_R1   |         128 |                              -0.00683594 | -0.027832   |  0.0146484  |       0.549045  | 0.717467    |
| neuron |      35 | neuron_L35_R1 |         128 |                              -0.00732422 | -0.0407776  |  0.0244141  |       0.677832  | 0.723021    |
| neuron |      30 | neuron_L30_R1 |         128 |                              -0.010498   | -0.027832   |  0.00610352 |       0.247175  | 0.359528    |
| neuron |      10 | neuron_L10_R1 |         128 |                              -0.019043   | -0.0380859  |  0          |       0.0661934 | 0.132387    |
| head   |      10 | head_L10_R1   |         128 |                              -0.0192871  | -0.0495667  |  0.0131836  |       0.228877  | 0.359528    |
| head   |       0 | head_L0_R1    |         128 |                              -0.0322266  | -0.0708008  |  0.00244141 |       0.0917908 | 0.163184    |

## Copy-task specificity vs random-transition control

| kind   |   layer | pair_id       |   n_copy_prompts |   n_control_prompts |   copy_minus_control_specificity |    ci95_low |   ci95_high |   permutation_p |   copy_selected_vs_random_mean |   control_selected_vs_random_mean |        q_bh |
|:-------|--------:|:--------------|-----------------:|--------------------:|---------------------------------:|------------:|------------:|----------------:|-------------------------------:|----------------------------------:|------------:|
| head   |      15 | head_L15_R1   |              128 |                 128 |                       0.262634   |  0.192596   |  0.342078   |       9.999e-05 |                     0.290283   |                        0.0276489  | 0.000319968 |
| neuron |       0 | neuron_L0_R1  |              128 |                 128 |                       0.243129   |  0.155785   |  0.338381   |       9.999e-05 |                     0.240723   |                       -0.00240636 | 0.000319968 |
| neuron |      25 | neuron_L25_R1 |              128 |                 128 |                       0.0312805  |  0.00616302 |  0.055909   |       0.0136986 |                     0.0234375  |                       -0.00784302 | 0.0313112   |
| head   |      30 | head_L30_R1   |              128 |                 128 |                       0.0221558  | -0.00622635 |  0.049044   |       0.120288  |                     0.00537109 |                       -0.0167847  | 0.213845    |
| head   |      10 | head_L10_R1   |              128 |                 128 |                       0.0158386  | -0.0233765  |  0.0534065  |       0.423958  |                    -0.0192871  |                       -0.0351257  | 0.565277    |
| neuron |       5 | neuron_L5_R1  |              128 |                 128 |                       0.014328   | -0.0200352  |  0.0560318  |       0.50065   |                     0.0119629  |                       -0.00236511 | 0.616185    |
| neuron |      30 | neuron_L30_R1 |              128 |                 128 |                       0.00318909 | -0.0200199  |  0.0274357  |       0.793221  |                    -0.010498   |                       -0.0136871  | 0.846102    |
| neuron |      20 | neuron_L20_R1 |              128 |                 128 |                      -0.00121307 | -0.0301514  |  0.0270777  |       0.931607  |                     0.00219727 |                        0.00341034 | 0.931607    |
| head   |      25 | head_L25_R1   |              128 |                 128 |                      -0.0148926  | -0.0413216  |  0.010743   |       0.275572  |                    -0.00683594 |                        0.00805664 | 0.400833    |
| head   |       0 | head_L0_R1    |              128 |                 128 |                      -0.0168381  | -0.0703123  |  0.0367365  |       0.545045  |                    -0.0322266  |                       -0.0153885  | 0.622909    |
| neuron |      10 | neuron_L10_R1 |              128 |                 128 |                      -0.021347   | -0.0507668  |  0.0081337  |       0.145985  |                    -0.019043   |                        0.00230408 | 0.233577    |
| head   |      20 | head_L20_R1   |              128 |                 128 |                      -0.0285797  | -0.060183   |  0.00430298 |       0.0883912 |                    -0.0378418  |                       -0.00926208 | 0.176782    |

## Activation patching

|   layer | patch_location           | kind   | selection      |   component |   n |   mean_margin_recovery |   margin_recovery_ci95_low |   margin_recovery_ci95_high |   mean_target_logit_recovery |   target_logit_recovery_ci95_low |   target_logit_recovery_ci95_high |
|--------:|:-------------------------|:-------|:---------------|------------:|----:|-----------------------:|---------------------------:|----------------------------:|-----------------------------:|---------------------------------:|----------------------------------:|
|       0 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.000821004 |               -0.00403423  |                  0.00592509 |                  0.00234236  |                     -0.00318088  |                       0.0106996   |
|       0 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.998109    |                0.97585     |                  1.02103    |                  0.993485    |                      0.974471    |                       1.00902     |
|       0 | head_context_final_query | head   | discovery_top  |          12 |  12 |           -0.000348834 |               -0.00642255  |                  0.00599477 |                 -0.00209183  |                     -0.00550522  |                       0.000751302 |
|       0 | head_context_final_query | head   | random_control |          14 |  12 |           -0.000866989 |               -0.00384018  |                  0.0023245  |                 -0.00404943  |                     -0.00697859  |                      -0.00115515  |
|       0 | mlp_neuron_final_query   | neuron | discovery_top  |        9146 |  12 |            0.00444148  |               -1.36776e-05 |                  0.00948932 |                 -0.0024802   |                     -0.00536007  |                       0.000450853 |
|       0 | mlp_neuron_final_query   | neuron | random_control |        8349 |  12 |            0.00219664  |               -0.00229468  |                  0.00663145 |                  0.000176223 |                     -0.00304159  |                       0.00345681  |
|       5 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.0104691   |               -0.00298455  |                  0.0243272  |                  0.00789302  |                     -0.00412483  |                       0.020918    |
|       5 | block_successor_position | block  | clean_patch    |         nan |  12 |            1.0205      |                0.979793    |                  1.06275    |                  1.00069     |                      0.959977    |                       1.04805     |
|       5 | head_context_final_query | head   | discovery_top  |           4 |  12 |            0.0044953   |               -0.0032293   |                  0.0139023  |                 -0.000138402 |                     -0.00189612  |                       0.00161816  |
|       5 | head_context_final_query | head   | random_control |          13 |  12 |           -0.0011576   |               -0.00702953  |                  0.003917   |                  0.00116202  |                     -0.00184245  |                       0.00412926  |
|       5 | mlp_neuron_final_query   | neuron | discovery_top  |        4801 |  12 |           -0.000145798 |               -0.00393639  |                  0.00415295 |                 -0.00137082  |                     -0.00369896  |                       0.00107751  |
|       5 | mlp_neuron_final_query   | neuron | random_control |         979 |  12 |           -0.00112187  |               -0.00783981  |                  0.00549748 |                 -0.00200586  |                     -0.00420124  |                       0.000224531 |
|      10 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.000111388 |               -0.0137812   |                  0.0139735  |                  0.00815781  |                     -0.00628556  |                       0.0238654   |
|      10 | block_successor_position | block  | clean_patch    |         nan |  12 |            1.01507     |                0.956       |                  1.09097    |                  0.952735    |                      0.897568    |                       1.00872     |
|      10 | head_context_final_query | head   | discovery_top  |           1 |  12 |            0.00247198  |               -0.0036961   |                  0.00871863 |                 -0.000898774 |                     -0.00504066  |                       0.00312333  |
|      10 | head_context_final_query | head   | random_control |           7 |  12 |            0.00133836  |               -0.00510481  |                  0.00782682 |                  8.73087e-05 |                     -0.00254924  |                       0.0025952   |
|      10 | mlp_neuron_final_query   | neuron | discovery_top  |        6073 |  12 |            0.00177773  |               -0.00386467  |                  0.00735245 |                 -0.00116155  |                     -0.00427588  |                       0.00175439  |
|      10 | mlp_neuron_final_query   | neuron | random_control |        1988 |  12 |            0.00430275  |               -0.00174842  |                  0.0110152  |                  0.00181985  |                     -0.000708377 |                       0.00449838  |
|      15 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00420069  |               -0.00838669  |                  0.0177431  |                  0.00560052  |                     -0.00555092  |                       0.0187373   |
|      15 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.962562    |                0.892183    |                  1.04512    |                  0.892246    |                      0.830178    |                       0.946879    |
|      15 | head_context_final_query | head   | discovery_top  |          11 |  12 |            0.0122197   |                0.00351951  |                  0.0203303  |                  0.00988762  |                      0.000192149 |                       0.0200227   |
|      15 | head_context_final_query | head   | random_control |           4 |  12 |           -0.00282469  |               -0.00743079  |                  0.00152778 |                  1.19593e-05 |                     -0.0029327   |                       0.00282603  |
|      15 | mlp_neuron_final_query   | neuron | discovery_top  |        7050 |  12 |           -0.00783634  |               -0.0143148   |                 -0.00102606 |                 -0.00161177  |                     -0.00546409  |                       0.00185715  |
|      15 | mlp_neuron_final_query   | neuron | random_control |        3622 |  12 |            0.00150834  |               -0.00338264  |                  0.00635647 |                 -0.00109609  |                     -0.00448591  |                       0.00216497  |
|      20 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00995149  |               -0.00238471  |                  0.0265736  |                  0.00796094  |                     -0.001384    |                       0.0182915   |
|      20 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.922831    |                0.84087     |                  1.00889    |                  0.844089    |                      0.773728    |                       0.907252    |
|      20 | head_context_final_query | head   | discovery_top  |           1 |  12 |            0.0034089   |               -0.000441436 |                  0.00753513 |                  0.00548356  |                      0.00217037  |                       0.00913382  |
|      20 | head_context_final_query | head   | random_control |          10 |  12 |            8.55427e-05 |               -0.00526797  |                  0.0054589  |                 -0.00124958  |                     -0.00423057  |                       0.00197749  |
|      20 | mlp_neuron_final_query   | neuron | discovery_top  |        2252 |  12 |           -0.000192739 |               -0.00254655  |                  0.00219373 |                  0.00111979  |                     -0.00113666  |                       0.00357014  |
|      20 | mlp_neuron_final_query   | neuron | random_control |        8712 |  12 |           -0.00200301  |               -0.00731387  |                  0.00263775 |                 -0.00092664  |                     -0.00250028  |                       0           |
|      25 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00730707  |                0.000428226 |                  0.0150665  |                  0.00934479  |                      0.00081362  |                       0.0186739   |
|      25 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.508356    |                0.436493    |                  0.581093   |                  0.448824    |                      0.359624    |                       0.54229     |
|      25 | head_context_final_query | head   | discovery_top  |           8 |  12 |            0.00277626  |               -0.00259974  |                  0.00872822 |                 -0.000990401 |                     -0.00388598  |                       0.00206823  |
|      25 | head_context_final_query | head   | random_control |           5 |  12 |            0.00832512  |                0.0036755   |                  0.0128839  |                  0.00456131  |                      0.00161031  |                       0.00841252  |
|      25 | mlp_neuron_final_query   | neuron | discovery_top  |        9433 |  12 |            0.00667979  |                0.000504644 |                  0.0125546  |                  0.000564471 |                     -0.00548185  |                       0.00577924  |
|      25 | mlp_neuron_final_query   | neuron | random_control |        1256 |  12 |           -0.00294345  |               -0.00836805  |                  0.00215486 |                  0.000665961 |                     -0.00197144  |                       0.00320171  |
|      30 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00948835  |                0.00241184  |                  0.0178873  |                  0.0109444   |                      0.00205927  |                       0.0214246   |
|      30 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.364977    |                0.271932    |                  0.470583   |                  0.289101    |                      0.192653    |                       0.396346    |
|      30 | head_context_final_query | head   | discovery_top  |           9 |  12 |            0.00391274  |               -0.00362807  |                  0.0111239  |                  0.00211388  |                     -0.00119547  |                       0.00574432  |
|      30 | head_context_final_query | head   | random_control |          12 |  12 |            0.00516303  |               -0.00168613  |                  0.0142463  |                  0.000553159 |                     -0.00307366  |                       0.0042017   |
|      30 | mlp_neuron_final_query   | neuron | discovery_top  |        8506 |  12 |            0.00161653  |               -0.00392597  |                  0.00690274 |                  0.00153549  |                     -0.00198396  |                       0.0048103   |
|      30 | mlp_neuron_final_query   | neuron | random_control |        7289 |  12 |            0.00171654  |               -0.00318904  |                  0.00654824 |                  0.000952861 |                     -0.0015711   |                       0.00382609  |
|      35 | block_random_position    | block  | clean_patch    |         nan |  12 |            0           |                0           |                  0          |                  0           |                      0           |                       0           |
|      35 | block_successor_position | block  | clean_patch    |         nan |  12 |            0           |                0           |                  0          |                  0           |                      0           |                       0           |
|      35 | head_context_final_query | head   | discovery_top  |           0 |  12 |           -0.103445    |               -0.152378    |                 -0.0566209  |                 -0.0831367   |                     -0.133507    |                      -0.0391287   |
|      35 | head_context_final_query | head   | random_control |          11 |  12 |            0.00168759  |               -0.00355227  |                  0.00655573 |                  0.00167821  |                     -0.00108734  |                       0.00443654  |
|      35 | mlp_neuron_final_query   | neuron | discovery_top  |         505 |  12 |            0.00929177  |               -0.00252937  |                  0.0224039  |                  0.0136932   |                     -0.000732858 |                       0.0326638   |
|      35 | mlp_neuron_final_query   | neuron | random_control |        7065 |  12 |            0.000284934 |               -0.00561929  |                  0.00667971 |                  0.00436611  |                      0.000452361 |                       0.0085543   |

## Interpretation guidance

A component is more specifically implicated in sequence copying when its selected-vs-control ablation effect is larger on copy prompts than on independent random-transition prompts, with uncertainty intervals and corrected tests supporting the difference.
The successor-B attention score is a candidate-discovery signal only; it is not proof of an induction circuit.
Component-specific patching and non-trivial position controls provide additional causal evidence, but should be interpreted alongside baseline performance and held-out ablations.
