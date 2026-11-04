"""Run reproducible TinyGPT component sweeps, induction probes and patching."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretability_lab.core import GPTConfig, CharTokenizer, TinyGPT, DEMO_TEXT, collect_mlp_hidden
from interpretability_lab.research import (
    make_induction_prompt, induction_score_table, evaluate_ablation_prompts,
    summarize_ablation_rows, activation_patch_sweep, sweep_components,
)
from interpretability_lab.experiment import seed_everything, environment_metadata, write_json


def load_tiny(device):
    ckpt = ROOT / "artifacts" / "tiny_gpt.pt"
    if not ckpt.exists():
        raise FileNotFoundError("Train first: python scripts/train_tiny.py --steps 1500")
    data = torch.load(ckpt, map_location=device, weights_only=False)
    tok = CharTokenizer(chars=data.get("chars", sorted(set(DEMO_TEXT))))
    model = TinyGPT(GPTConfig(**data["config"])).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, tok, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=8, help="Maximum number of prompts used from the built-in suite (currently 8)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-neurons-per-layer", type=int, default=4)
    args = parser.parse_args()
    if args.prompts < 3 or args.top_neurons_per_layer < 0:
        parser.error("prompts must be >= 3 and top-neurons-per-layer must be >= 0")
    seed_everything(args.seed)
    model, tok, checkpoint = load_tiny(args.device)
    texts = [
        "the quick brown fox", "transformers process sequences", "a model predicts the next token",
        "mechanistic interpretability studies internal computation", "attention uses previous context",
        "the model reads tokens and builds representations", "a neural network transforms input sequences",
        "causal tests measure changes in model predictions",
    ]
    # Keep only prompts fully represented in this checkpoint's character vocabulary.
    texts = [t for t in texts if all(c in tok.stoi for c in t)]
    if len(texts) < 3:
        raise RuntimeError("Checkpoint tokenizer does not cover the built-in prompt suite")
    texts = texts[: min(args.prompts, len(texts))]
    prompts = [torch.tensor([tok.encode(t)[-model.config.block_size:]], device=args.device) for t in texts]
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)

    # Component sweeps across every head and a discovery-selected subset of MLP neurons.
    sweep_rows, sweep_summary = [], []
    for layer in range(model.config.n_layer):
        rows, summary = sweep_components(model, prompts, "head", layer)
        sweep_rows.extend(rows)
        sweep_summary.extend(summary)
    if args.top_neurons_per_layer:
        activity = {layer: [] for layer in range(model.config.n_layer)}
        for ids in prompts:
            for layer, acts in enumerate(collect_mlp_hidden(model, ids)):
                activity[layer].append(acts.abs().mean(dim=(0, 1)))
        for layer in range(model.config.n_layer):
            mean_activity = torch.stack(activity[layer]).mean(0)
            selected = torch.topk(mean_activity, min(args.top_neurons_per_layer, mean_activity.numel())).indices.tolist()
            rows, summary = sweep_components(model, prompts, "neuron", layer, selected)
            sweep_rows.extend(rows)
            sweep_summary.extend(summary)
    pd.DataFrame(sweep_rows).to_csv(out / "tinygpt_component_ablation_per_prompt.csv", index=False)
    pd.DataFrame(sweep_summary).sort_values("kl_mean", ascending=False).to_csv(
        out / "tinygpt_component_ablation_summary.csv", index=False
    )

    # Preserve the original focused head-0 report for backwards compatibility.
    focused = evaluate_ablation_prompts(model, prompts, "head", 0, 0)
    focused_summary = summarize_ablation_rows(focused)
    pd.DataFrame(focused).to_csv(out / "multi_prompt_head_ablation.csv", index=False)
    write_json(out / "multi_prompt_head_ablation_summary.json", focused_summary)

    # Repeated-token probes are controlled synthetic diagnostics, not circuit proof.
    vocab = tok.vocab_size
    candidates = [i for i in range(vocab) if tok.decode([i]).isalnum()]
    probes = []
    for i in range(min(args.prompts, max(1, len(candidates) - 5))):
        indices = [(i + j) % len(candidates) for j in range(6)]
        token_ids = [candidates[k] for k in indices]
        if len(set(token_ids)) != 6:
            continue
        a, b, *fillers = token_ids
        probes.append(make_induction_prompt(a, b, fillers, device=args.device))
    if probes:
        score = induction_score_table(model, probes)
        pd.DataFrame(score).to_csv(out / "induction_attention_scores.csv", index_label="layer")

    # Position-aligned clean/corrupted pair; only the second character is changed.
    clean = prompts[0]
    corrupt = clean.clone()
    replacement = (int(clean[0, 1]) + 1) % tok.vocab_size
    corrupt[0, 1] = replacement
    with torch.no_grad():
        clean_logits, _ = model(clean)
        target = int(clean_logits[0, -1].argmax())
    patch = activation_patch_sweep(model, clean, corrupt, target)
    write_json(out / "activation_patch_demo.json", patch)

    metadata = environment_metadata(args.seed, "TinyGPT", texts)
    metadata.update({
        "checkpoint_training": checkpoint.get("training", {}),
        "model_config": model.config.__dict__,
        "component_sweep": "all attention heads and top-activity MLP neurons per layer",
        "neuron_discovery_and_evaluation_overlap": True,
        "limitations": [
            "TinyGPT uses a small character-level demonstration corpus and is not representative of GPT-2.",
            "Neuron selection and evaluation use the same prompt set in this quick suite; use held-out prompts for confirmatory work.",
            "Attention scores and ablation sensitivity alone do not establish a semantic circuit.",
        ],
    })
    write_json(out / "tinygpt_research_run_metadata.json", metadata)
    print(f"Saved component sweeps and diagnostics to {out}")
    print(pd.DataFrame(sweep_summary).sort_values("kl_mean", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
