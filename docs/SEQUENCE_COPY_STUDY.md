# Controlled sequence-copying study

## Research question

How do attention heads and MLP intermediate coordinates causally contribute to a decoder-only Transformer's ability to continue a repeated-token sequence, and how robust are candidate mechanisms across models?

## Task construction

Each fixed-length nine-token prompt has the form:

`[P0 ... Pk, A, B, F0 ... Fm, A] -> predict B`

The prefix length varies from one to three tokens. The number of filler tokens is adjusted so every prompt has length nine. Within a prompt, the eight sequence token IDs plus the corruption token are distinct; sampled token IDs are also not reused across prompts within a split. Discovery and held-out evaluation use disjoint token-identity pools. The corrupted prompt replaces the earlier successor `B` with a distinct token `X` while preserving sequence length and every other position.

The task is a controlled *probe*, not a claim that arbitrary token sequences are a natural-language capability. A model's baseline held-out copy accuracy must be reported before interpreting intervention effects.

## Experimental protocol

1. **Discovery attention scan.** For each selected layer and head, measure attention from the final repeated `A` query to the first `A` key. Rank candidates by mean attention lift over other previous positions. This is descriptive evidence only.
2. **Discovery MLP scan.** Rank intermediate coordinates by mean absolute post-gating activation at the final query position.
3. **Held-out causal evaluation.** Ablate the selected head/neuron only at the final query position. Pair each discovery-ranked candidate with a random component from the same layer. Measure target-token logit margin drop, target probability change, KL divergence and copy-accuracy changes on held-out prompts.
4. **Paired statistics.** Compare candidate ablation effects with same-layer random controls on the same prompts. Report percentile-bootstrap 95% confidence intervals, paired sign-flip permutation p-values and Benjamini-Hochberg q-values over the component-pair family.
5. **Activation patching.** Corrupt the earlier successor token `B` to `X`. For each selected layer, patch the clean block output back into the corrupted run at (a) the corrupted successor position and (b) the final query position. Report target-logit and target-margin recovery. These are complementary interventions, not interchangeable proof of a full circuit.
6. **Cross-model validation.** Run the identical seed, prompt-generation protocol and evaluation settings for GPT-2 and SmolLM3-3B-Base. Model-specific head indices are not assumed to correspond.

## Recommended first runs

Start with GPT-2 on a GPU or CPU to check the full workflow:

```bash
python scripts/run_sequence_copy_study.py --model-id gpt2 --device auto --discovery-cases 96 --evaluation-cases 128 --patch-cases 12 --layers 8 --batch-size 8
```

Then repeat with the larger model on a GPU with sufficient memory:

```bash
python scripts/run_sequence_copy_study.py --model-id HuggingFaceTB/SmolLM3-3B-Base --device cuda --dtype bfloat16 --discovery-cases 96 --evaluation-cases 128 --patch-cases 12 --layers 8 --batch-size 4
```

Increase the number of selected layers and candidates only after the first complete run succeeds. For a final report, aim for at least 100 held-out prompts and repeat the study with at least three seeds. Example GPT-2 repeats:

```bash
python scripts/run_sequence_copy_study.py --model-id gpt2 --seed 7
python scripts/run_sequence_copy_study.py --model-id gpt2 --seed 13
python scripts/run_sequence_copy_study.py --model-id gpt2 --seed 29
```

Use matching case counts, top-per-layer settings, selected-layer counts and patch counts for cross-model studies. The model token IDs are intentionally tokenizer-specific and are not directly aligned across models.

## Interpretation rules

- If held-out baseline copy accuracy is near chance, component effects should not be described as supporting a robust copying mechanism.
- High attention to the earlier `A` is a ranking signal, not a functional label.
- A larger candidate-ablation effect than a same-layer random control is evidence of relative sensitivity, not by itself proof of a complete circuit.
- Prefer conclusions that triangulate candidate ranking, held-out ablation, activation patching and controls.
- Report all selected components, not only the strongest one. Include uncertainty intervals and adjusted q-values.
- If GPT-2 and SmolLM3 disagree, report the difference rather than forcing a shared interpretation.

## Generated artifacts

The runner writes a timestamped folder containing the exact decoded and token-ID prompts, discovery attention scores and summaries, neuron rankings, candidate/control selections, held-out baseline metrics and bootstrap summary, per-prompt ablation results, component summaries, candidate-versus-control statistics, activation-patching results and bootstrap summary, run metadata and a generated report.
