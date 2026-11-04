# Sequence-copying mechanistic interpretability study — Protocol V2

- Model: `gpt2`
- Architecture: `GPT2LMHeadModel`
- Selected layers: `[0, 2, 3, 5, 6, 8, 9, 11]`
- Discovery/copy-eval/control cases: `96/128/128`
- Patch cases: `12`
- Held-out copy accuracy: **0.5391**
- Random-transition control top-1 accuracy: **0.0000**

## Held-out baseline uncertainty

| metric             |   n |      mean |      std |   ci95_low |   ci95_high |
|:-------------------|----:|----------:|---------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.539062 | 0.50043  |  0.453125  |    0.625    |
| target_probability | 128 |  0.100332 | 0.149969 |  0.0758609 |    0.127133 |
| copy_margin        | 128 | -0.148336 | 2.46976  | -0.583977  |    0.276054 |

## Candidate vs random-control ablation effects on copy task

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |        q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|------------:|
| head   |       6 | head_L6_R1    |         128 |                               0.384113   |  0.246486   |  0.536117   |      9.999e-05  | 0.000228549 |
| neuron |       3 | neuron_L3_R1  |         128 |                               0.354395   |  0.271769   |  0.446729   |      9.999e-05  | 0.000228549 |
| neuron |       6 | neuron_L6_R1  |         128 |                               0.188073   |  0.134385   |  0.249387   |      9.999e-05  | 0.000228549 |
| head   |       9 | head_L9_R1    |         128 |                               0.0800103  |  0.0349448  |  0.127329   |      0.00079992 | 0.00159984  |
| head   |       8 | head_L8_R1    |         128 |                               0.0632809  |  0.00815895 |  0.123912   |      0.0424958  | 0.056661    |
| neuron |      11 | neuron_L11_R1 |         128 |                               0.0396776  |  0.0271745  |  0.0529719  |      9.999e-05  | 0.000228549 |
| neuron |       9 | neuron_L9_R1  |         128 |                               0.0158153  | -0.02341    |  0.0551434  |      0.432157   | 0.460967    |
| neuron |       2 | neuron_L2_R1  |         128 |                              -0.00544205 | -0.0255705  |  0.0158129  |      0.612239   | 0.612239    |
| neuron |       0 | neuron_L0_R1  |         128 |                              -0.00801438 | -0.0139499  | -0.00267251 |      0.00639936 | 0.00930816  |
| neuron |       8 | neuron_L8_R1  |         128 |                              -0.0129865  | -0.0313535  |  0.00563272 |      0.175782   | 0.216348    |
| neuron |       5 | neuron_L5_R1  |         128 |                              -0.0133105  | -0.0332123  |  0.00630435 |      0.193081   | 0.220664    |
| head   |       5 | head_L5_R1    |         128 |                              -0.125263   | -0.208414   | -0.0450034  |      0.00389961 | 0.00693264  |

## Copy-task specificity vs random-transition control

| kind   |   layer | pair_id       |   n_copy_prompts |   n_control_prompts |   copy_minus_control_specificity |   ci95_low |   ci95_high |   permutation_p |   copy_selected_vs_random_mean |   control_selected_vs_random_mean |       q_bh |
|:-------|--------:|:--------------|-----------------:|--------------------:|---------------------------------:|-----------:|------------:|----------------:|-------------------------------:|----------------------------------:|-----------:|
| head   |       6 | head_L6_R1    |              128 |                 128 |                       0.450745   |  0.303912  |  0.60309    |      9.999e-05  |                     0.384113   |                      -0.0666319   | 0.00017776 |
| neuron |       3 | neuron_L3_R1  |              128 |                 128 |                       0.357807   |  0.273165  |  0.449771   |      9.999e-05  |                     0.354395   |                      -0.00341231  | 0.00017776 |
| neuron |       9 | neuron_L9_R1  |              128 |                 128 |                       0.162042   |  0.108607  |  0.217224   |      9.999e-05  |                     0.0158153  |                      -0.146227    | 0.00017776 |
| neuron |       6 | neuron_L6_R1  |              128 |                 128 |                       0.122023   |  0.0641787 |  0.184003   |      9.999e-05  |                     0.188073   |                       0.0660499   | 0.00017776 |
| head   |       9 | head_L9_R1    |              128 |                 128 |                       0.0794899  |  0.0317548 |  0.127981   |      0.00129987 |                     0.0800103  |                       0.000520408 | 0.00207979 |
| neuron |      11 | neuron_L11_R1 |              128 |                 128 |                       0.0433295  |  0.0253711 |  0.0623108  |      9.999e-05  |                     0.0396776  |                      -0.00365198  | 0.00017776 |
| head   |       8 | head_L8_R1    |              128 |                 128 |                       0.0227612  | -0.0386087 |  0.087266   |      0.494951   |                     0.0632809  |                       0.0405197   | 0.565658   |
| neuron |       2 | neuron_L2_R1  |              128 |                 128 |                       0.00535271 | -0.0173433 |  0.028733   |      0.645335   |                    -0.00544205 |                      -0.0107948   | 0.645335   |
| neuron |       0 | neuron_L0_R1  |              128 |                 128 |                      -0.00204104 | -0.0102004 |  0.00643836 |      0.625837   |                    -0.00801438 |                      -0.00597334  | 0.645335   |
| neuron |       8 | neuron_L8_R1  |              128 |                 128 |                      -0.0268335  | -0.0487169 | -0.00455075 |      0.0188981  |                    -0.0129865  |                       0.013847    | 0.0251975  |
| neuron |       5 | neuron_L5_R1  |              128 |                 128 |                      -0.0520625  | -0.0774901 | -0.0277016  |      9.999e-05  |                    -0.0133105  |                       0.0387521   | 0.00017776 |
| head   |       5 | head_L5_R1    |              128 |                 128 |                      -0.0567219  | -0.161246  |  0.0485756  |      0.29987    |                    -0.125263   |                      -0.0685415   | 0.369071   |

## Activation patching

|   layer | patch_location           | kind   | selection      |   component |   n |   mean_margin_recovery |   margin_recovery_ci95_low |   margin_recovery_ci95_high |   mean_target_logit_recovery |   target_logit_recovery_ci95_low |   target_logit_recovery_ci95_high |
|--------:|:-------------------------|:-------|:---------------|------------:|----:|-----------------------:|---------------------------:|----------------------------:|-----------------------------:|---------------------------------:|----------------------------------:|
|       0 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00769095  |                0.000726709 |                 0.015783    |                  0.0610525   |                     -0.000360269 |                       0.173568    |
|       0 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.988824    |                0.964674    |                 1.0136      |                  1.33758     |                      1.00125     |                       1.95117     |
|       0 | head_context_final_query | head   | discovery_top  |           3 |  12 |            2.36422e-05 |               -7.35091e-06 |                 6.45266e-05 |                 -8.62347e-05 |                     -0.000322515 |                       7.10825e-05 |
|       0 | head_context_final_query | head   | random_control |          10 |  12 |            0.00398817  |               -0.00279562  |                 0.0129983   |                 -0.00421235  |                     -0.0194162   |                       0.00824376  |
|       0 | mlp_neuron_final_query   | neuron | discovery_top  |        1207 |  12 |           -3.45521e-05 |               -0.000252503 |                 0.000123142 |                  0.00522691  |                     -0.000723404 |                       0.0161014   |
|       0 | mlp_neuron_final_query   | neuron | random_control |         442 |  12 |            7.15385e-05 |                7.97706e-06 |                 0.000142577 |                 -0.000603329 |                     -0.00133927  |                      -4.28491e-05 |
|       2 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.0182886   |               -0.00949974  |                 0.0618896   |                  0.16522     |                      0.00274818  |                       0.444518    |
|       2 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.969474    |                0.926693    |                 1.01142     |                  1.15218     |                      0.826066    |                       1.61359     |
|       2 | head_context_final_query | head   | discovery_top  |           7 |  12 |           -0.00166416  |               -0.00279437  |                -0.000617734 |                 -0.0162395   |                     -0.0477765   |                       0.00145325  |
|       2 | head_context_final_query | head   | random_control |          10 |  12 |           -0.000732116 |               -0.00251846  |                 0.000772531 |                 -0.0684656   |                     -0.210019    |                       0.00548933  |
|       2 | mlp_neuron_final_query   | neuron | discovery_top  |        1971 |  12 |            0.000722345 |               -0.00017248  |                 0.00162224  |                 -0.0197097   |                     -0.0565387   |                       0.00169681  |
|       2 | mlp_neuron_final_query   | neuron | random_control |         920 |  12 |           -5.75205e-06 |               -7.60763e-05 |                 7.62591e-05 |                 -0.000425794 |                     -0.00149132  |                       0.000385242 |
|       3 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00986943  |               -0.0422664   |                 0.0652118   |                  0.59204     |                     -0.000267433 |                       1.7496      |
|       3 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.960803    |                0.915308    |                 1.00761     |                  0.855058    |                      0.593476    |                       1.05756     |
|       3 | head_context_final_query | head   | discovery_top  |          10 |  12 |           -0.0019415   |               -0.00497713  |                 0.000852027 |                 -0.0112532   |                     -0.0490668   |                       0.0141053   |
|       3 | head_context_final_query | head   | random_control |           4 |  12 |            0.001671    |               -0.000450004 |                 0.00474395  |                 -0.107659    |                     -0.311273    |                      -0.00171151  |
|       3 | mlp_neuron_final_query   | neuron | discovery_top  |        1718 |  12 |           -0.00081624  |               -0.002355    |                 0.000475761 |                 -0.00966643  |                     -0.0254458   |                       0.00318252  |
|       3 | mlp_neuron_final_query   | neuron | random_control |         176 |  12 |           -0.00126521  |               -0.00420355  |                 0.000753349 |                  0.00228361  |                     -0.00787203  |                       0.0139068   |
|       5 | block_random_position    | block  | clean_patch    |         nan |  12 |           -0.00309072  |               -0.0543909   |                 0.0393816   |                  0.882871    |                     -0.0109659   |                       2.63332     |
|       5 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.912557    |                0.845372    |                 0.980994    |                 -0.0322043   |                     -1.95072     |                       1.04592     |
|       5 | head_context_final_query | head   | discovery_top  |           5 |  12 |            0.0485794   |                0.0296065   |                 0.0708137   |                  0.357513    |                     -0.0153911   |                       1.06589     |
|       5 | head_context_final_query | head   | random_control |           3 |  12 |            0.00499188  |               -0.00205945  |                 0.0146061   |                  0.0754873   |                      0.00877215  |                       0.184501    |
|       5 | mlp_neuron_final_query   | neuron | discovery_top  |        1691 |  12 |            0.000834156 |               -0.000341975 |                 0.00237154  |                 -0.00837421  |                     -0.0287253   |                       0.00371738  |
|       5 | mlp_neuron_final_query   | neuron | random_control |        1854 |  12 |           -0.000336748 |               -0.00108955  |                 0.000285665 |                 -0.00878934  |                     -0.0239137   |                      -0.000194365 |
|       6 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.0077206   |               -0.0222727   |                 0.0414095   |                  0.416287    |                     -0.00452257  |                       1.21205     |
|       6 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.673529    |                0.584084    |                 0.756053    |                 -0.0980211   |                     -1.3243      |                       0.657593    |
|       6 | head_context_final_query | head   | discovery_top  |           9 |  12 |            0.128324    |                0.0754435   |                 0.187926    |                  0.00558837  |                     -0.276453    |                       0.182091    |
|       6 | head_context_final_query | head   | random_control |           7 |  12 |            0.00294988  |               -0.000592023 |                 0.00679215  |                 -0.0797396   |                     -0.219135    |                      -4.95434e-05 |
|       6 | mlp_neuron_final_query   | neuron | discovery_top  |         624 |  12 |           -0.00302155  |               -0.00665654  |                 0.00033335  |                  0.0789333   |                     -0.000161559 |                       0.219309    |
|       6 | mlp_neuron_final_query   | neuron | random_control |        1462 |  12 |            0.00244599  |               -0.000946586 |                 0.0077065   |                 -0.0196369   |                     -0.0546731   |                      -0.000838714 |
|       8 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00925819  |               -0.00735126  |                 0.0270893   |                  0.188136    |                      0.00633552  |                       0.505439    |
|       8 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.462721    |                0.378479    |                 0.557875    |                 -0.0265004   |                     -0.803676    |                       0.451339    |
|       8 | head_context_final_query | head   | discovery_top  |           1 |  12 |            0.0213417   |                0.00249062  |                 0.0403207   |                  0.0425647   |                     -0.0273715   |                       0.125056    |
|       8 | head_context_final_query | head   | random_control |           4 |  12 |            0.000848119 |               -0.00327181  |                 0.00574823  |                  0.136777    |                     -0.00161501  |                       0.406572    |
|       8 | mlp_neuron_final_query   | neuron | discovery_top  |        2038 |  12 |            0.00848712  |                0.00199667  |                 0.0156293   |                  0.032277    |                      0.00394784  |                       0.0766834   |
|       8 | mlp_neuron_final_query   | neuron | random_control |         803 |  12 |            0.000166481 |               -0.000226349 |                 0.000606437 |                 -0.000567701 |                     -0.00352057  |                       0.00237862  |
|       9 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00894536  |               -0.00429049  |                 0.0218571   |                  0.212709    |                      0.00517817  |                       0.581379    |
|       9 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.394207    |                0.308286    |                 0.493232    |                  0.326261    |                      0.176827    |                       0.490437    |
|       9 | head_context_final_query | head   | discovery_top  |           6 |  12 |            0.0609114   |                0.0377619   |                 0.0844351   |                  0.281121    |                      0.058518    |                       0.665094    |
|       9 | head_context_final_query | head   | random_control |           9 |  12 |            0.0683285   |                0.0536503   |                 0.0823903   |                  0.119378    |                     -0.0039665   |                       0.332519    |
|       9 | mlp_neuron_final_query   | neuron | discovery_top  |         840 |  12 |            0.000406976 |               -0.00100857  |                 0.00189585  |                  0.0218503   |                     -0.00384767  |                       0.0535292   |
|       9 | mlp_neuron_final_query   | neuron | random_control |        2722 |  12 |            1.14918e-05 |               -0.000227879 |                 0.000257336 |                  0.000521365 |                     -0.000548879 |                       0.00192606  |
|      11 | block_random_position    | block  | clean_patch    |         nan |  12 |            0           |                0           |                 0           |                  0           |                      0           |                       0           |
|      11 | block_successor_position | block  | clean_patch    |         nan |  12 |            0           |                0           |                 0           |                  0           |                      0           |                       0           |
|      11 | head_context_final_query | head   | discovery_top  |          10 |  12 |           -0.0585525   |               -0.0966217   |                -0.0293726   |                 -0.016458    |                     -0.0418476   |                       0.00821928  |
|      11 | head_context_final_query | head   | random_control |           7 |  12 |            0.0383566   |                0.00010806  |                 0.102311    |                  0.330635    |                     -0.0394362   |                       0.984844    |
|      11 | mlp_neuron_final_query   | neuron | discovery_top  |        2600 |  12 |            0.00237345  |                0.000528277 |                 0.00472178  |                 -0.165526    |                     -0.561653    |                       0.0666308   |
|      11 | mlp_neuron_final_query   | neuron | random_control |        2032 |  12 |           -0.00132011  |               -0.00409186  |                 0.00117784  |                 -0.00332813  |                     -0.0150163   |                       0.00599192  |

## Interpretation guidance

A component is more specifically implicated in sequence copying when its selected-vs-control ablation effect is larger on copy prompts than on independent random-transition prompts, with uncertainty intervals and corrected tests supporting the difference.
The successor-B attention score is a candidate-discovery signal only; it is not proof of an induction circuit.
Component-specific patching and non-trivial position controls provide additional causal evidence, but should be interpreted alongside baseline performance and held-out ablations.
