# SmolLM3-3B-Base — Protocol V2, three-seed findings summary

**Evidence status:** This document transcribes and organises the project author's consolidated summary of the three SmolLM3 Protocol V2 runs (seeds 7, 13 and 29). The raw per-run V2 CSV bundles were not attached in the files available when this final ZIP was assembled. Treat the numerical ranges below as author-reported results pending a raw-artifact audit; do not treat this file as a substitute for the individual run reports and CSVs.

## Study configuration

- Model: `HuggingFaceTB/SmolLM3-3B-Base`
- Protocol: V2
- Seeds: 7, 13, 29
- Discovery / held-out copy evaluation / random-transition control: 96 / 128 / 128 cases per run
- Activation-patching prompts: 12 per run
- Layers screened: 8 of the model's 36 layers
- Architecture recorded during runtime validation: 36 layers, 16 query heads, 4 KV heads, MLP width 11,008

## Baseline behaviour

| Seed | Held-out copy accuracy (author-reported) |
|---:|---:|
| 7 | 0.44 |
| 13 | 0.46 |
| 29 | 0.46 |

The reported confidence intervals overlap substantially. Mean target probability was approximately 5–7%, and mean copy margins were negative (approximately −0.5 to −1.0). Random-transition control top-1 accuracy was 0% in all three runs.

Interpretation: the model exhibits partial, not strong, sequence-copying performance on this synthetic probe. A 0% random-transition accuracy is a useful sanity check, but this control is not a natural-language benchmark and may have a floor effect.

## Replicated attention-head candidate: Layer 15, Head 11

The author reports that L15 H11 was selected during discovery in all three seeds. Its ablation produced an additional copy-task mean-margin drop of approximately 0.29–0.31 relative to a matched random head (BH-adjusted q < 0.001), with copy-minus-control specificity of approximately +0.26 to +0.32 (BH-adjusted q < 0.001).

This supports describing L15 H11 as a **candidate attention head that causally contributes to the measured sequence-copying behaviour**. It does not establish that the head alone implements a complete induction circuit. The reported top-1 prediction changed on only about 6% of prompts, with accuracy moving from about 0.44 to 0.39 under ablation.

## Other attention-head observations

- L5 H4 was selected across seeds and had a consistent negative selected-versus-random effect of approximately −0.10 to −0.11. Its ablation affected the random-transition control more than the copy task; interpret it as a general sensitivity component, not a copy-specific head.
- L10 H1 was selected in all three seeds, but its ablation effect was small and not statistically significant.

## MLP neuron observations

Some neuron coordinates were selected repeatedly (including L0 9146, L10 6073, L25 9433, L30 8506 and L35 505 in all three seeds; L15 7050 and L20 2209 in two seeds), but their causal effects did not replicate consistently. No MLP neuron should currently be claimed as copy-specific.

L35 neuron 505 was reported to have unusually high mean activity (about 17 versus about 5 for its matched control). Its ablation may be off-manifold; exclude it from copy-circuit claims pending investigation.

## Activation-patching observations

- Whole-block patching at the earlier successor position reportedly recovered approximately 0.9–1.0 of the copy-margin gap in early layers through about layer 20, falling to approximately 0.2–0.5 around layers 25–30.
- The random-position control was approximately zero.
- Layer 35 recovery was zero and should not be interpreted as mechanistic evidence.
- Seed 13's layer 5–20 recovery was reported lower (approximately 0.62–0.86) than seeds 7 and 29 (approximately 0.90–1.0); the reason is unresolved.
- Component-specific head/neuron patches had margin recovery approximately within ±0.01 at layers 0–30. This is consistent with distributed computation or redundancy, but does not distinguish those explanations.
- Patching used only 12 prompts per seed, so intervals are imprecise.

## Limitations and next research questions

1. Only one top candidate per layer was tested (`top_per_layer = 1`), and only 8 of 36 layers were screened.
2. The successor-B attention score is a discovery signal, not proof of an induction-head function.
3. The random-transition control is a generic sensitivity control, not a difficulty-matched natural-text task; its 0% accuracy warrants caution.
4. Ablation can move activations off the model's normal manifold.
5. Patching uses only 12 prompts, limiting precision.
6. BH correction was performed within each run; a pooled or meta-analytic analysis across seeds would strengthen the overall claim.
7. Layer/head/neuron indices must not be compared directly across GPT-2 and SmolLM3 architectures.

Recommended extensions: screen top 3–5 heads in layers 12–18; test combined ablations; perform path patching from L15 H11 to downstream components; increase patching cases; and add a natural-text induction benchmark with a behaviourally matched control.
