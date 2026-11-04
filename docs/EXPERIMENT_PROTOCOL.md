# Experiment protocol

## A. Multi-prompt component ablation

1. Freeze the model checkpoint and tokenizer. Record their identifiers / hashes.
2. Define prompts before inspection. Use a discovery set for component selection and a separate held-out evaluation set for confirmation.
3. For each prompt, record the baseline next-token distribution.
4. Intervene on one component at a time: zero the selected attention head output or MLP post-activation coordinate.
5. Record KL divergence, baseline-argmax token change, and the baseline-argmax logit shift for every prompt.
6. Aggregate by component using prompt-level observations. Report sample count, mean, standard deviation, median and percentile-bootstrap 95% CI.
7. Include all prompt rows, including small and null effects. Do not treat component ranking as a functional explanation.

## B. Activation patching

1. Choose a clean and corrupted prompt pair with equal tokenized lengths.
2. Define the target token from the clean run before examining patched outputs.
3. Cache clean and corrupted residual activations at each block output.
4. For each layer, patch the clean activation at the aligned position into the corrupted run.
5. Also patch the corrupted activation into the corrupted run as a self-patch control.
6. Record clean, corrupted, self-patch and clean-patched logits for the fixed target token.
7. Compute normalized logit recovery without clipping. A value outside [0, 1] is possible and should be discussed.

## C. Repeated-token attention probe

1. Construct a sequence with a repeated token and distinguishable intervening tokens.
2. Record attention from a later occurrence to the earlier occurrence by layer and head.
3. Compare against other positions / prompts and inspect stability.
4. Treat this only as a pattern screen. Confirm any proposed mechanism with causal interventions and held-out tests.

## D. Statistical and reporting rules

- The bootstrap resampling unit is the prompt, not the number of components.
- Confidence intervals describe the sampled prompt distribution and are not proof of generalization to all contexts.
- Avoid selecting and evaluating candidate components on the same prompts for confirmatory claims.
- Report model revision, tokenizer, prompt source, seed, environment, command, all exclusions and all controls.
- Attention maps and PCA plots are descriptive views, not causal evidence.


## E. Controlled sequence-copying study

Use `docs/SEQUENCE_COPY_STUDY.md` as the primary confirmatory protocol. It defines a fixed-length repeated-token probe, a disjoint discovery/evaluation vocabulary split, ranked candidates, same-layer random controls, held-out final-query interventions, paired prompt-level statistics and clean-to-corrupted activation patching. Preserve `sequence_copy_cases.csv` alongside result files so every reported metric can be traced back to the exact token IDs and decoded input. The model baseline must be reported before component effects are interpreted.
