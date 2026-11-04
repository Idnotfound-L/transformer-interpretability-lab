# Sequence-copying mechanistic interpretability study — Protocol V2

- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Architecture: `SmolLM3ForCausalLM`
- Selected layers: `[0, 5, 10, 15, 20, 25, 30, 35]`
- Discovery/copy-eval/control cases: `96/128/128`
- Patch cases: `12`
- Held-out copy accuracy: **0.4375**
- Random-transition control top-1 accuracy: **0.0000**

## Held-out baseline uncertainty

| metric             |   n |       mean |       std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|----------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.4375    | 0.498028  |  0.351562  |   0.523438  |
| target_probability | 128 |  0.0487995 | 0.0674119 |  0.0376295 |   0.0611044 |
| copy_margin        | 128 | -0.988525  | 3.0689    | -1.53395   |  -0.460919  |

## Candidate vs random-control ablation effects on copy task

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |        q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|------------:|
| head   |      15 | head_L15_R1   |         128 |                              0.308838    |  0.218256   |  0.401855   |      9.999e-05  | 0.000319968 |
| neuron |      20 | neuron_L20_R1 |         128 |                              0.0627441   |  0.041748   |  0.0837402  |      9.999e-05  | 0.000319968 |
| neuron |       5 | neuron_L5_R1  |         128 |                              0.0559082   |  0.0180603  |  0.101318   |      0.00279972 | 0.00639936  |
| neuron |      25 | neuron_L25_R1 |         128 |                              0.0175781   | -0.00317383 |  0.0378418  |      0.10219    | 0.181671    |
| neuron |      10 | neuron_L10_R1 |         128 |                              0.0148926   | -0.00585938 |  0.0351562  |      0.177982   | 0.258883    |
| head   |       0 | head_L0_R1    |         128 |                              0.012207    | -0.046875   |  0.0612793  |      0.690631   | 0.789292    |
| neuron |       0 | neuron_L0_R1  |         128 |                              0.00878906  | -0.0219727  |  0.0424866  |      0.624838   | 0.769031    |
| head   |      30 | head_L30_R1   |         128 |                              0.0012207   | -0.019043   |  0.0219727  |      0.924808   | 0.964104    |
| neuron |      30 | neuron_L30_R1 |         128 |                              0.000732422 | -0.0195312  |  0.020752   |      0.964104   | 0.964104    |
| head   |      20 | head_L20_R1   |         128 |                             -0.0100098   | -0.0349121  |  0.0144043  |      0.451555   | 0.602073    |
| neuron |      15 | neuron_L15_R1 |         128 |                             -0.019043    | -0.0422363  |  0.00512695 |      0.131187   | 0.209899    |
| head   |      10 | head_L10_R1   |         128 |                             -0.0437012   | -0.0791077  | -0.0102478  |      0.0154985  | 0.0309969   |

## Copy-task specificity vs random-transition control

| kind   |   layer | pair_id       |   n_copy_prompts |   n_control_prompts |   copy_minus_control_specificity |     ci95_low |   ci95_high |   permutation_p |   copy_selected_vs_random_mean |   control_selected_vs_random_mean |        q_bh |
|:-------|--------:|:--------------|-----------------:|--------------------:|---------------------------------:|-------------:|------------:|----------------:|-------------------------------:|----------------------------------:|------------:|
| head   |      15 | head_L15_R1   |              128 |                 128 |                      0.321289    |  0.221403    |   0.420902  |      9.999e-05  |                    0.308838    |                       -0.0124512  | 0.000319968 |
| neuron |       5 | neuron_L5_R1  |              128 |                 128 |                      0.0592346   |  0.0180305   |   0.107606  |      0.00429957 |                    0.0559082   |                       -0.00332642 | 0.00859914  |
| neuron |      20 | neuron_L20_R1 |              128 |                 128 |                      0.0498352   |  0.0229469   |   0.0775162 |      0.00019998 |                    0.0627441   |                        0.0129089  | 0.00053328  |
| neuron |      30 | neuron_L30_R1 |              128 |                 128 |                      0.0270081   | -0.000244904 |   0.0544136 |      0.0532947  |                    0.000732422 |                       -0.0262756  | 0.0852715   |
| neuron |       0 | neuron_L0_R1  |              128 |                 128 |                      0.0185699   | -0.0230114   |   0.0625156 |      0.414659   |                    0.00878906  |                       -0.00978088 | 0.60314     |
| head   |       0 | head_L0_R1    |              128 |                 128 |                      0.00578308  | -0.0578621   |   0.0645782 |      0.861914   |                    0.012207    |                        0.00642395 | 0.973603    |
| neuron |      25 | neuron_L25_R1 |              128 |                 128 |                      0.000915527 | -0.0252098   |   0.0257278 |      0.943306   |                    0.0175781   |                        0.0166626  | 0.973603    |
| head   |      30 | head_L30_R1   |              128 |                 128 |                     -0.000488281 | -0.0260017   |   0.0252075 |      0.973603   |                    0.0012207   |                        0.00170898 | 0.973603    |
| neuron |      10 | neuron_L10_R1 |              128 |                 128 |                     -0.000579834 | -0.0261543   |   0.0275887 |      0.969503   |                    0.0148926   |                        0.0154724  | 0.973603    |
| head   |      10 | head_L10_R1   |              128 |                 128 |                     -0.00971985  | -0.050967    |   0.0309612 |      0.651835   |                   -0.0437012   |                       -0.0339813  | 0.869113    |
| head   |      20 | head_L20_R1   |              128 |                 128 |                     -0.0499573   | -0.0898445   |  -0.0133667 |      0.0106989  |                   -0.0100098   |                        0.0399475  | 0.0190203   |
| neuron |      15 | neuron_L15_R1 |              128 |                 128 |                     -0.0583801   | -0.0884399   |  -0.0263672 |      0.00049995 |                   -0.019043    |                        0.0393372  | 0.00114274  |

## Activation patching

|   layer | patch_location           | kind   | selection      |   component |   n |   mean_margin_recovery |   margin_recovery_ci95_low |   margin_recovery_ci95_high |   mean_target_logit_recovery |   target_logit_recovery_ci95_low |   target_logit_recovery_ci95_high |
|--------:|:-------------------------|:-------|:---------------|------------:|----:|-----------------------:|---------------------------:|----------------------------:|-----------------------------:|---------------------------------:|----------------------------------:|
|       0 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00312741  |                -0.00305279 |                 0.0112076   |                  0.00221356  |                     -0.00209521  |                       0.00658776  |
|       0 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.989853    |                 0.981503   |                 0.998594    |                  0.978603    |                      0.968014    |                       0.989634    |
|       0 | head_context_final_query | head   | discovery_top  |          11 |  12 |           -0.00399497  |                -0.00704303 |                -0.000397781 |                 -0.00568737  |                     -0.00850257  |                      -0.0030304   |
|       0 | head_context_final_query | head   | random_control |          12 |  12 |            0.00109435  |                -0.00299995 |                 0.00569941  |                  0.000249354 |                     -0.00248732  |                       0.00313623  |
|       0 | mlp_neuron_final_query   | neuron | discovery_top  |        9146 |  12 |           -0.0025874   |                -0.00684382 |                 0.000814732 |                 -0.000569022 |                     -0.00313934  |                       0.00187553  |
|       0 | mlp_neuron_final_query   | neuron | random_control |        1766 |  12 |            0.000568894 |                -0.00197368 |                 0.00343961  |                 -0.00038407  |                     -0.00299621  |                       0.00273021  |
|       5 | block_random_position    | block  | clean_patch    |         nan |  12 |           -0.00667491  |                -0.0132186  |                -0.00141243  |                 -0.000747812 |                     -0.00633279  |                       0.00531424  |
|       5 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.988443    |                 0.930655   |                 1.04645     |                  0.956051    |                      0.901713    |                       1.01517     |
|       5 | head_context_final_query | head   | discovery_top  |           4 |  12 |            0.000470074 |                -0.0044293  |                 0.00564726  |                 -0.00118467  |                     -0.00497565  |                       0.00237105  |
|       5 | head_context_final_query | head   | random_control |           3 |  12 |           -0.000973366 |                -0.00469902 |                 0.00333266  |                 -0.0010306   |                     -0.00545623  |                       0.00289454  |
|       5 | mlp_neuron_final_query   | neuron | discovery_top  |        6086 |  12 |           -0.00230144  |                -0.00780769 |                 0.00284869  |                  0.000470155 |                     -0.00269584  |                       0.00402749  |
|       5 | mlp_neuron_final_query   | neuron | random_control |        9286 |  12 |           -0.00175017  |                -0.00497753 |                 0.00122583  |                 -0.000505465 |                     -0.00330274  |                       0.00231599  |
|      10 | block_random_position    | block  | clean_patch    |         nan |  12 |           -0.00447997  |                -0.0191233  |                 0.0099548   |                 -0.00348302  |                     -0.0122973   |                       0.00511143  |
|      10 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.97008     |                 0.898479   |                 1.03981     |                  0.927819    |                      0.822226    |                       1.0237      |
|      10 | head_context_final_query | head   | discovery_top  |           1 |  12 |           -0.000224144 |                -0.00507112 |                 0.00457317  |                 -0.00261087  |                     -0.0075052   |                       0.0024482   |
|      10 | head_context_final_query | head   | random_control |          12 |  12 |           -0.00454493  |                -0.0101672  |                 0.00144309  |                 -0.00180625  |                     -0.00685135  |                       0.00328131  |
|      10 | mlp_neuron_final_query   | neuron | discovery_top  |        6073 |  12 |           -0.00482363  |                -0.0102473  |                 0.000307625 |                 -0.00187661  |                     -0.00620615  |                       0.00289326  |
|      10 | mlp_neuron_final_query   | neuron | random_control |        3749 |  12 |           -0.000827794 |                -0.00519099 |                 0.00343809  |                 -0.0012201   |                     -0.00597198  |                       0.00303711  |
|      15 | block_random_position    | block  | clean_patch    |         nan |  12 |           -0.0131387   |                -0.0340391  |                 0.00569133  |                 -0.00222827  |                     -0.0109832   |                       0.00770475  |
|      15 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.945229    |                 0.849693   |                 1.02291     |                  0.885648    |                      0.766439    |                       0.975843    |
|      15 | head_context_final_query | head   | discovery_top  |          11 |  12 |            0.000862669 |                -0.0140271  |                 0.0153833   |                  0.00122736  |                     -0.00382955  |                       0.00695706  |
|      15 | head_context_final_query | head   | random_control |           7 |  12 |           -0.00368916  |                -0.00683925 |                -0.000913236 |                 -0.0027022   |                     -0.00549145  |                       0.000313038 |
|      15 | mlp_neuron_final_query   | neuron | discovery_top  |        7050 |  12 |           -0.00050525  |                -0.00705121 |                 0.006501    |                 -6.81606e-05 |                     -0.00466107  |                       0.00494401  |
|      15 | mlp_neuron_final_query   | neuron | random_control |        9753 |  12 |            0.00123954  |                -0.00146127 |                 0.00441877  |                  0.00165225  |                     -0.00126056  |                       0.00499137  |
|      20 | block_random_position    | block  | clean_patch    |         nan |  12 |           -0.00503509  |                -0.0251223  |                 0.0157807   |                  0.00675828  |                     -0.00611331  |                       0.0207697   |
|      20 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.903589    |                 0.799687   |                 0.986666    |                  0.857725    |                      0.745464    |                       0.944391    |
|      20 | head_context_final_query | head   | discovery_top  |           1 |  12 |            0.00415532  |                -0.00283231 |                 0.0109907   |                  0.00158835  |                     -0.002175    |                       0.00596064  |
|      20 | head_context_final_query | head   | random_control |          13 |  12 |           -0.00276358  |                -0.0077051  |                 0.00138042  |                  0.00145127  |                     -0.000432617 |                       0.00348735  |
|      20 | mlp_neuron_final_query   | neuron | discovery_top  |        2209 |  12 |           -0.00129849  |                -0.00368092 |                 0.000833423 |                 -0.00129623  |                     -0.00382124  |                       0.00123153  |
|      20 | mlp_neuron_final_query   | neuron | random_control |        8756 |  12 |            0.00180749  |                -0.00390001 |                 0.0080904   |                  0.00102278  |                     -0.00412181  |                       0.00670732  |
|      25 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.0015727   |                -0.0138505  |                 0.0199016   |                  0.00834104  |                     -0.00272336  |                       0.0217196   |
|      25 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.47191     |                 0.35637    |                 0.58653     |                  0.440857    |                      0.322827    |                       0.550568    |
|      25 | head_context_final_query | head   | discovery_top  |           8 |  12 |            0.000149452 |                -0.0109913  |                 0.0133386   |                  0.00419532  |                     -0.00358748  |                       0.013363    |
|      25 | head_context_final_query | head   | random_control |           2 |  12 |            0.0102182   |                 0.00433369 |                 0.0159743   |                  0.00919895  |                      0.00412232  |                       0.0140122   |
|      25 | mlp_neuron_final_query   | neuron | discovery_top  |        9433 |  12 |            0.00118312  |                -0.00377297 |                 0.00656802  |                 -0.00263336  |                     -0.00726106  |                       0.00252263  |
|      25 | mlp_neuron_final_query   | neuron | random_control |       10926 |  12 |           -3.90888e-06 |                -0.00655915 |                 0.00608613  |                 -0.00238185  |                     -0.00655888  |                       0.00176723  |
|      30 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00807535  |                -0.00781569 |                 0.0275552   |                  0.0120605   |                     -8.81082e-05 |                       0.028119    |
|      30 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.334947    |                 0.231464   |                 0.46507     |                  0.288393    |                      0.182568    |                       0.40932     |
|      30 | head_context_final_query | head   | discovery_top  |           6 |  12 |            0.00782212  |                -0.00849732 |                 0.0260591   |                  0.0086373   |                     -0.00182582  |                       0.0210979   |
|      30 | head_context_final_query | head   | random_control |           7 |  12 |            0.00485516  |                -0.00286343 |                 0.013533    |                  0.00611937  |                     -0.000970176 |                       0.0137211   |
|      30 | mlp_neuron_final_query   | neuron | discovery_top  |        8506 |  12 |           -0.00174242  |                -0.00712912 |                 0.00326588  |                 -0.000635212 |                     -0.00454899  |                       0.00338586  |
|      30 | mlp_neuron_final_query   | neuron | random_control |        1407 |  12 |            0.00232209  |                -0.0013002  |                 0.00575312  |                 -0.000276609 |                     -0.00399285  |                       0.00359095  |
|      35 | block_random_position    | block  | clean_patch    |         nan |  12 |            0           |                 0          |                 0           |                  0           |                      0           |                       0           |
|      35 | block_successor_position | block  | clean_patch    |         nan |  12 |            0           |                 0          |                 0           |                  0           |                      0           |                       0           |
|      35 | head_context_final_query | head   | discovery_top  |           0 |  12 |           -0.124309    |                -0.166837   |                -0.0800966   |                 -0.114346    |                     -0.172693    |                      -0.0602916   |
|      35 | head_context_final_query | head   | random_control |           7 |  12 |           -0.0108571   |                -0.016722   |                -0.00539836  |                 -0.00682529  |                     -0.0117698   |                      -0.00219528  |
|      35 | mlp_neuron_final_query   | neuron | discovery_top  |         505 |  12 |            0.00159372  |                -0.00728503 |                 0.0106204   |                  0.00138039  |                     -0.0230116   |                       0.0238748   |
|      35 | mlp_neuron_final_query   | neuron | random_control |        9554 |  12 |           -0.00314796  |                -0.0124094  |                 0.00457571  |                 -0.00210335  |                     -0.0163721   |                       0.0077432   |

## Interpretation guidance

A component is more specifically implicated in sequence copying when its selected-vs-control ablation effect is larger on copy prompts than on independent random-transition prompts, with uncertainty intervals and corrected tests supporting the difference.
The successor-B attention score is a candidate-discovery signal only; it is not proof of an induction circuit.
Component-specific patching and non-trivial position controls provide additional causal evidence, but should be interpreted alongside baseline performance and held-out ablations.
