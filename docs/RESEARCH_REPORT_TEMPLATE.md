# Research note: [question]

## Abstract
[One paragraph: question, model, intervention, prompt set, measured result, limitations.]

## Research question and hypotheses
- Research question:
- Primary hypothesis:
- Null / alternative explanations:
- Predefined primary metric:

## Model and data
- Model identifier / checkpoint hash:
- Tokenizer:
- Corpus or prompt-source and licence:
- Prompt inclusion / exclusion rules:
- Discovery prompt count:
- Held-out evaluation prompt count:
- Random seed(s):
- Software versions and hardware:

## Experimental protocol
- Component / layer / head / neuron:
- Intervention implementation:
- Clean and corrupted inputs:
- Token-position alignment method:
- Baseline:
- Negative / self-patch controls:
- Number of repeated runs:

## Results
| Component | N prompts | Mean KL | SD | Bootstrap 95% CI | Top-token change rate |
|---|---:|---:|---:|---:|---:|
| | | | | | |

Attach the full per-prompt CSV. Include null, negative and inconsistent effects.

## Interpretation
[Separate observation from mechanistic interpretation. State competing explanations.]

## Limitations
[Prompt coverage, model scope, component selection bias, metric limitations, tokenizer alignment, compute constraints.]

## Reproducibility
- Command:
- Git commit:
- Run metadata JSON:
- Artifact filenames:
- Deviations from protocol:

## Conclusion and next experiment
[What is justified by the evidence, and what would falsify the interpretation?]


## Controlled sequence-copying study addendum

- Task definition: `[prefix, A, B, fillers, A] -> predict B`
- Number of discovery and evaluation prompts; were token IDs disjoint across prompts and splits?
- Baseline held-out copy accuracy with bootstrap 95% CI:
- Top discovery-ranked attention heads and their attention-lift scores:
- Discovery-ranked MLP coordinates and activation scores:
- Candidate-vs-same-layer-random margin-drop differences, 95% CIs, permutation p-values and BH q-values:
- Activation-patching results at the corrupted successor position and final query position:
- Do results replicate across seeds?
- Does each model show sufficient baseline task competence to interpret interventions?
- Which observations are descriptive, which are causal sensitivities, and which circuit claims remain unsupported?
