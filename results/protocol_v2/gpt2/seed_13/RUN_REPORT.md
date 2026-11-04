# Sequence-copying mechanistic interpretability study — Protocol V2

- Model: `gpt2`
- Architecture: `GPT2LMHeadModel`
- Selected layers: `[0, 2, 3, 5, 6, 8, 9, 11]`
- Discovery/copy-eval/control cases: `96/128/128`
- Patch cases: `12`
- Held-out copy accuracy: **0.5000**
- Random-transition control top-1 accuracy: **0.0000**

## Held-out baseline uncertainty

| metric             |   n |       mean |      std |   ci95_low |   ci95_high |
|:-------------------|----:|-----------:|---------:|-----------:|------------:|
| copy_accuracy      | 128 |  0.5       | 0.501965 |  0.414062  |    0.585938 |
| target_probability | 128 |  0.0760265 | 0.132514 |  0.0538143 |    0.101057 |
| copy_margin        | 128 | -0.521049  | 2.36818  | -0.921761  |   -0.113252 |

## Candidate vs random-control ablation effects on copy task

| kind   |   layer | pair_id       |   n_prompts |   selected_minus_random_mean_margin_drop |    ci95_low |   ci95_high |   permutation_p |       q_bh |
|:-------|--------:|:--------------|------------:|-----------------------------------------:|------------:|------------:|----------------:|-----------:|
| head   |       2 | head_L2_R1    |         128 |                               0.28192    | -0.0216727  |  0.585079   |      0.0708929  | 0.0872528  |
| head   |       6 | head_L6_R1    |         128 |                               0.271069   |  0.143275   |  0.408962   |      0.00029997 | 0.00053328 |
| neuron |       3 | neuron_L3_R1  |         128 |                               0.196691   |  0.145787   |  0.250498   |      9.999e-05  | 0.00019998 |
| head   |       0 | head_L0_R1    |         128 |                               0.183424   |  0.0961114  |  0.275607   |      9.999e-05  | 0.00019998 |
| head   |       8 | head_L8_R1    |         128 |                               0.167657   |  0.120415   |  0.213627   |      9.999e-05  | 0.00019998 |
| neuron |       6 | neuron_L6_R1  |         128 |                               0.0938074  |  0.0602556  |  0.131477   |      9.999e-05  | 0.00019998 |
| head   |       9 | head_L9_R1    |         128 |                               0.0880692  |  0.050627   |  0.129472   |      9.999e-05  | 0.00019998 |
| neuron |       5 | neuron_L5_R1  |         128 |                               0.0130047  | -0.0111722  |  0.036524   |      0.292771   | 0.312289   |
| neuron |       8 | neuron_L8_R1  |         128 |                               0.0112262  | -0.00327966 |  0.027154   |      0.152085   | 0.173811   |
| neuron |      11 | neuron_L11_R1 |         128 |                               0.00919506 |  0.00252231 |  0.0159195  |      0.00919908 | 0.0133805  |
| neuron |       0 | neuron_L0_R1  |         128 |                               0.00476149 |  9.0535e-05 |  0.00969244 |      0.0524948  | 0.069993   |
| neuron |       9 | neuron_L9_R1  |         128 |                               0.00268343 | -0.0298578  |  0.0365623  |      0.866713   | 0.866713   |

## Copy-task specificity vs random-transition control

| kind   |   layer | pair_id       |   n_copy_prompts |   n_control_prompts |   copy_minus_control_specificity |    ci95_low |   ci95_high |   permutation_p |   copy_selected_vs_random_mean |   control_selected_vs_random_mean |       q_bh |
|:-------|--------:|:--------------|-----------------:|--------------------:|---------------------------------:|------------:|------------:|----------------:|-------------------------------:|----------------------------------:|-----------:|
| head   |       2 | head_L2_R1    |              128 |                 128 |                       0.545101   |  0.170856   |   0.913169  |      0.00329967 |                     0.28192    |                       -0.263181   | 0.00527947 |
| head   |       6 | head_L6_R1    |              128 |                 128 |                       0.359618   |  0.228359   |   0.502026  |      9.999e-05  |                     0.271069   |                       -0.0885481  | 0.00019998 |
| neuron |       3 | neuron_L3_R1  |              128 |                 128 |                       0.230125   |  0.171436   |   0.289035  |      9.999e-05  |                     0.196691   |                       -0.0334341  | 0.00019998 |
| head   |       8 | head_L8_R1    |              128 |                 128 |                       0.183368   |  0.127216   |   0.239444  |      9.999e-05  |                     0.167657   |                       -0.0157108  | 0.00019998 |
| neuron |       9 | neuron_L9_R1  |              128 |                 128 |                       0.151124   |  0.100979   |   0.202874  |      9.999e-05  |                     0.00268343 |                       -0.14844    | 0.00019998 |
| head   |       9 | head_L9_R1    |              128 |                 128 |                       0.125379   |  0.0788173  |   0.173901  |      9.999e-05  |                     0.0880692  |                       -0.0373101  | 0.00019998 |
| neuron |       6 | neuron_L6_R1  |              128 |                 128 |                       0.0731412  |  0.0373648  |   0.110936  |      9.999e-05  |                     0.0938074  |                        0.0206661  | 0.00019998 |
| head   |       0 | head_L0_R1    |              128 |                 128 |                       0.0630588  | -0.142291   |   0.294801  |      0.590241   |                     0.183424   |                        0.120366   | 0.674561   |
| neuron |       5 | neuron_L5_R1  |              128 |                 128 |                       0.0216135  | -0.00735743 |   0.0517761 |      0.161884   |                     0.0130047  |                       -0.00860885 | 0.215845   |
| neuron |      11 | neuron_L11_R1 |              128 |                 128 |                       0.0160705  |  0.00561191 |   0.0268348 |      0.00329967 |                     0.00919506 |                       -0.00687549 | 0.00527947 |
| neuron |       8 | neuron_L8_R1  |              128 |                 128 |                       0.00316191 | -0.0134947  |   0.0209516 |      0.721428   |                     0.0112262  |                        0.00806433 | 0.721428   |
| neuron |       0 | neuron_L0_R1  |              128 |                 128 |                       0.00204816 | -0.00635954 |   0.0109329 |      0.641036   |                     0.00476149 |                        0.00271332 | 0.683772   |

## Activation patching

|   layer | patch_location           | kind   | selection      |   component |   n |   mean_margin_recovery |   margin_recovery_ci95_low |   margin_recovery_ci95_high |   mean_target_logit_recovery |   target_logit_recovery_ci95_low |   target_logit_recovery_ci95_high |
|--------:|:-------------------------|:-------|:---------------|------------:|----:|-----------------------:|---------------------------:|----------------------------:|-----------------------------:|---------------------------------:|----------------------------------:|
|       0 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00048903  |               -0.00732116  |                 0.00656411  |                 -0.121002    |                     -0.534015    |                       0.138635    |
|       0 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.984471    |                0.970411    |                 0.99764     |                  1.2461      |                      0.609018    |                       2.16195     |
|       0 | head_context_final_query | head   | discovery_top  |           9 |  12 |           -0.0037016   |               -0.00796138  |                 0.000445925 |                 -0.0820088   |                     -0.278564    |                       0.0392131   |
|       0 | head_context_final_query | head   | random_control |           5 |  12 |            4.34372e-06 |               -6.20529e-08 |                 1.22329e-05 |                 -8.78708e-06 |                     -2.50808e-05 |                       3.00632e-06 |
|       0 | mlp_neuron_final_query   | neuron | discovery_top  |        1207 |  12 |           -0.000357547 |               -0.000831243 |                 1.62591e-05 |                 -0.00211761  |                     -0.0106221   |                       0.0040303   |
|       0 | mlp_neuron_final_query   | neuron | random_control |        1126 |  12 |            8.44827e-05 |               -9.55049e-06 |                 0.000232142 |                 -0.00157642  |                     -0.00396457  |                       8.80976e-05 |
|       2 | block_random_position    | block  | clean_patch    |         nan |  12 |           -0.00433177  |               -0.0251095   |                 0.00781503  |                 -0.00874755  |                     -0.167672    |                       0.148568    |
|       2 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.963984    |                0.924438    |                 0.999548    |                  2.73133     |                      0.86496     |                       6.25773     |
|       2 | head_context_final_query | head   | discovery_top  |          10 |  12 |            0.000951214 |                7.80538e-05 |                 0.0018896   |                  0.0332454   |                     -0.00386398  |                       0.103566    |
|       2 | head_context_final_query | head   | random_control |           9 |  12 |           -0.000143061 |               -0.00126422  |                 0.00113451  |                 -0.0199016   |                     -0.0630812   |                       0.00506365  |
|       2 | mlp_neuron_final_query   | neuron | discovery_top  |        1971 |  12 |            0.000237797 |               -0.000226936 |                 0.000874179 |                 -0.00375766  |                     -0.0198486   |                       0.00693432  |
|       2 | mlp_neuron_final_query   | neuron | random_control |        2643 |  12 |            2.60366e-05 |               -0.00011249  |                 0.000175269 |                  0.0123714   |                     -0.00548882  |                       0.0443613   |
|       3 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00341853  |               -0.00704161  |                 0.0128862   |                  0.14198     |                     -0.0129093   |                       0.374783    |
|       3 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.971791    |                0.928966    |                 1.01294     |                  3.02225     |                      0.85918     |                       7.08592     |
|       3 | head_context_final_query | head   | discovery_top  |           6 |  12 |           -0.000194626 |               -0.00108732  |                 0.00059755  |                 -0.05081     |                     -0.14275     |                      -0.000223922 |
|       3 | head_context_final_query | head   | random_control |           9 |  12 |            0.00120528  |               -7.19394e-05 |                 0.00297702  |                  0.00784703  |                     -0.0046041   |                       0.0249221   |
|       3 | mlp_neuron_final_query   | neuron | discovery_top  |        1718 |  12 |           -0.000332523 |               -0.000997871 |                 0.000197406 |                 -0.00076851  |                     -0.0124816   |                       0.00672271  |
|       3 | mlp_neuron_final_query   | neuron | random_control |         614 |  12 |           -0.00026672  |               -0.000803607 |                 0.000175718 |                  0.00706694  |                      0.00195239  |                       0.013402    |
|       5 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.0104658   |                0.000490087 |                 0.021999    |                  0.283553    |                      0.0156942   |                       0.693383    |
|       5 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.887195    |                0.826242    |                 0.952904    |                  1.05997     |                      0.619843    |                       1.60209     |
|       5 | head_context_final_query | head   | discovery_top  |           5 |  12 |            0.0509308   |                0.0350324   |                 0.0667723   |                  0.937885    |                     -0.15665     |                       2.86328     |
|       5 | head_context_final_query | head   | random_control |           6 |  12 |           -0.000362453 |               -0.00108758  |                 0.000263202 |                  0.00627016  |                     -0.0070703   |                       0.0203464   |
|       5 | mlp_neuron_final_query   | neuron | discovery_top  |        2723 |  12 |           -0.000227819 |               -0.00140795  |                 0.000854445 |                  0.00624189  |                     -0.0104444   |                       0.0190033   |
|       5 | mlp_neuron_final_query   | neuron | random_control |        2201 |  12 |           -0.000627992 |               -0.00201466  |                 0.000837447 |                  0.16195     |                      0.00873567  |                       0.442027    |
|       6 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00843653  |               -0.00241688  |                 0.021143    |                  0.217584    |                      0.0135516   |                       0.504757    |
|       6 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.723673    |                0.630931    |                 0.813409    |                 -0.709609    |                     -3.09837     |                       0.661922    |
|       6 | head_context_final_query | head   | discovery_top  |           9 |  12 |            0.137542    |                0.0921123   |                 0.186511    |                  2.36285     |                      0.0462238   |                       6.74461     |
|       6 | head_context_final_query | head   | random_control |           0 |  12 |            0.00599387  |                0.00158966  |                 0.0111024   |                 -0.0840948   |                     -0.215258    |                      -0.00657295  |
|       6 | mlp_neuron_final_query   | neuron | discovery_top  |         624 |  12 |            0.0105936   |                0.00021457  |                 0.0227859   |                  0.0937065   |                      0.0134787   |                       0.18053     |
|       6 | mlp_neuron_final_query   | neuron | random_control |        2909 |  12 |           -0.000708303 |               -0.00188878  |                 0.000280692 |                  0.0136516   |                     -0.00846293  |                       0.0520876   |
|       8 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00862627  |               -0.000730828 |                 0.0191286   |                  0.209835    |                      0.00711442  |                       0.52258     |
|       8 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.417899    |                0.323763    |                 0.518093    |                 -0.231596    |                     -0.766344    |                       0.165842    |
|       8 | head_context_final_query | head   | discovery_top  |           6 |  12 |            0.0306178   |                0.0098254   |                 0.0565308   |                  0.0702542   |                     -0.0418788   |                       0.251819    |
|       8 | head_context_final_query | head   | random_control |           5 |  12 |            0.00993272  |                0.00451072  |                 0.0164232   |                  0.0221007   |                     -0.118209    |                       0.190982    |
|       8 | mlp_neuron_final_query   | neuron | discovery_top  |        2038 |  12 |            0.00498363  |                0.0010279   |                 0.00988758  |                 -0.0693434   |                     -0.257454    |                       0.050898    |
|       8 | mlp_neuron_final_query   | neuron | random_control |        1938 |  12 |            0.000353291 |               -0.000167211 |                 0.00092317  |                  0.00214751  |                     -0.00610574  |                       0.0131132   |
|       9 | block_random_position    | block  | clean_patch    |         nan |  12 |            0.00799852  |               -0.00105015  |                 0.0163775   |                  0.190924    |                     -0.00954252  |                       0.476179    |
|       9 | block_successor_position | block  | clean_patch    |         nan |  12 |            0.323664    |                0.247477    |                 0.402244    |                 -1.43892     |                     -4.21256     |                       0.112078    |
|       9 | head_context_final_query | head   | discovery_top  |           9 |  12 |            0.0646799   |                0.0469857   |                 0.0826576   |                 -0.0090413   |                     -0.213632    |                       0.146797    |
|       9 | head_context_final_query | head   | random_control |           3 |  12 |            0.0156304   |                0.00947622  |                 0.0221052   |                 -0.125874    |                     -0.355676    |                       0.00526263  |
|       9 | mlp_neuron_final_query   | neuron | discovery_top  |         840 |  12 |            0.000674145 |               -0.000619706 |                 0.00204746  |                 -0.0582414   |                     -0.167552    |                       0.0168133   |
|       9 | mlp_neuron_final_query   | neuron | random_control |        1329 |  12 |            6.28242e-05 |               -0.000877774 |                 0.000820675 |                  0.0173555   |                     -0.0195909   |                       0.0648      |
|      11 | block_random_position    | block  | clean_patch    |         nan |  12 |            0           |                0           |                 0           |                  0           |                      0           |                       0           |
|      11 | block_successor_position | block  | clean_patch    |         nan |  12 |            0           |                0           |                 0           |                  0           |                      0           |                       0           |
|      11 | head_context_final_query | head   | discovery_top  |          10 |  12 |           -0.0547809   |               -0.0839979   |                -0.0263494   |                  1.15392     |                     -0.0316094   |                       3.4815      |
|      11 | head_context_final_query | head   | random_control |           0 |  12 |            0.000502993 |               -0.0046054   |                 0.00474025  |                  0.382176    |                     -0.0749129   |                       1.23903     |
|      11 | mlp_neuron_final_query   | neuron | discovery_top  |        2600 |  12 |            0.000770209 |               -0.00088453  |                 0.0027922   |                  0.879303    |                      0.0251433   |                       2.43164     |
|      11 | mlp_neuron_final_query   | neuron | random_control |        2342 |  12 |           -0.000464633 |               -0.00162106  |                 0.000651929 |                  0.0386448   |                     -0.0536836   |                       0.145661    |

## Interpretation guidance

A component is more specifically implicated in sequence copying when its selected-vs-control ablation effect is larger on copy prompts than on independent random-transition prompts, with uncertainty intervals and corrected tests supporting the difference.
The successor-B attention score is a candidate-discovery signal only; it is not proof of an induction circuit.
Component-specific patching and non-trivial position controls provide additional causal evidence, but should be interpreted alongside baseline performance and held-out ablations.
