"""Run multi-prompt GPT-2 Small causal experiments with TransformerLens.

The script exports per-prompt observations and prompt-level bootstrap confidence
intervals. It is intentionally explicit about the fact that ablation sensitivity
is not, by itself, evidence of a semantic circuit.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretability_lab.experiment import (  # noqa: E402
    bootstrap_mean, environment_metadata, seed_everything, write_json,
)

DEFAULT_PROMPTS = [
    "The capital of France is", "The capital of Italy is", "A triangle has three",
    "A week has seven", "The opposite of hot is", "Water freezes at zero degrees",
    "The largest planet in the solar system is", "The first month of the year is",
    "The animal known for barking is", "The color of a clear daytime sky is",
    "A person writes with a", "The sun rises in the",
]
REPEATED_PROMPTS = [
    " red blue green yellow red", " cat dog bird fish cat",
    " one two three four one", " apple pear plum grape apple",
    " north south east west north", " spring summer autumn winter spring",
]


def _dist_metrics(base_logits: torch.Tensor, altered_logits: torch.Tensor) -> dict[str, float | int]:
    """Distribution and baseline-target metrics from next-token logits."""
    p_log = torch.log_softmax(base_logits.float(), dim=-1)
    q_log = torch.log_softmax(altered_logits.float(), dim=-1)
    p, q = p_log.exp(), q_log.exp()
    base_top = int(p.argmax().item())
    return {
        "kl_baseline_to_intervened": float((p * (p_log - q_log)).sum().item()),
        "baseline_top_token_id": base_top,
        "intervened_top_token_id": int(q.argmax().item()),
        "top_token_changed": int(base_top != int(q.argmax().item())),
        "baseline_top_probability": float(p.max().item()),
        "intervened_baseline_top_probability": float(q[base_top].item()),
        "baseline_top_logit_change": float((altered_logits.float()[base_top] - base_logits.float()[base_top]).item()),
    }


def _summarize_groups(frame: pd.DataFrame, group_columns: list[str], metric: str, seed: int, samples: int) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        stats = bootstrap_mean(group[metric].astype(float).tolist(), seed=seed, samples=samples)
        row.update({f"{metric}_{key}": value for key, value in stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _run_head_ablations(model, prompts: list[str], layer_count: int, seed: int, samples: int, device: str):
    rows: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt).to(device)
        baseline = model(tokens)[0][0, -1].float()
        for layer in range(layer_count):
            for head in range(model.cfg.n_heads):
                def zero_head(z, hook, selected=head):
                    edited = z.clone()
                    edited[:, :, selected, :] = 0
                    return edited
                intervened = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", zero_head)],
                )[0, -1].float()
                row = {
                    "prompt_id": prompt_index, "prompt": prompt,
                    "layer": layer, "head": head,
                }
                row.update(_dist_metrics(baseline, intervened))
                rows.append(row)
    per_prompt = pd.DataFrame(rows)
    summary = _summarize_groups(per_prompt, ["layer", "head"], "kl_baseline_to_intervened", seed, samples)
    top_change = per_prompt.groupby(["layer", "head"], as_index=False)["top_token_changed"].mean()
    target_logit = per_prompt.groupby(["layer", "head"], as_index=False)["baseline_top_logit_change"].mean()
    summary = summary.merge(top_change, on=["layer", "head"]).merge(target_logit, on=["layer", "head"])
    summary = summary.sort_values("kl_baseline_to_intervened_mean", ascending=False)
    return per_prompt, summary


def _find_candidate_neurons(model, prompts: list[str], layer_count: int, top_k: int, device: str):
    """Select highly active MLP dimensions on the declared prompt set (discovery set)."""
    if top_k <= 0:
        return {}
    totals = {layer: None for layer in range(layer_count)}
    counts = 0
    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(device)
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith(".mlp.hook_post"))
        for layer in range(layer_count):
            key = f"blocks.{layer}.mlp.hook_post"
            acts = cache[key][0].detach().float().abs().mean(dim=0).cpu()
            totals[layer] = acts if totals[layer] is None else totals[layer] + acts
        counts += 1
        del cache
    selected = {}
    for layer, values in totals.items():
        if values is not None:
            k = min(top_k, values.numel())
            selected[layer] = torch.topk(values / max(counts, 1), k=k).indices.tolist()
    return selected


def _run_neuron_ablations(model, prompts: list[str], candidates: dict[int, list[int]], seed: int, samples: int, device: str):
    rows: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt).to(device)
        baseline = model(tokens)[0][0, -1].float()
        for layer, neurons in candidates.items():
            for neuron in neurons:
                def zero_neuron(post, hook, selected=neuron):
                    edited = post.clone()
                    edited[:, :, selected] = 0
                    return edited
                intervened = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", zero_neuron)],
                )[0, -1].float()
                row = {
                    "prompt_id": prompt_index, "prompt": prompt,
                    "layer": layer, "neuron": neuron,
                }
                row.update(_dist_metrics(baseline, intervened))
                rows.append(row)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    per_prompt = pd.DataFrame(rows)
    summary = _summarize_groups(per_prompt, ["layer", "neuron"], "kl_baseline_to_intervened", seed, samples)
    top_change = per_prompt.groupby(["layer", "neuron"], as_index=False)["top_token_changed"].mean()
    target_logit = per_prompt.groupby(["layer", "neuron"], as_index=False)["baseline_top_logit_change"].mean()
    summary = summary.merge(top_change, on=["layer", "neuron"]).merge(target_logit, on=["layer", "neuron"])
    return per_prompt, summary.sort_values("kl_baseline_to_intervened_mean", ascending=False)


def _run_repeated_token_probe(model, prompts: list[str], device: str) -> pd.DataFrame:
    rows = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(device)
        ids = tokens[0].tolist()
        repeated_pair = None
        for query_pos in range(1, len(ids)):
            earlier = [key_pos for key_pos in range(query_pos) if ids[key_pos] == ids[query_pos]]
            if earlier:
                repeated_pair = (earlier[0], query_pos, ids[query_pos])
                break
        if repeated_pair is None:
            continue
        first_pos, query_pos, token_id = repeated_pair
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith("attn.hook_pattern"))
        for layer in range(model.cfg.n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]
            for head in range(model.cfg.n_heads):
                rows.append({
                    "prompt": prompt, "repeated_token_id": token_id,
                    "first_position": first_pos, "query_position": query_pos,
                    "layer": layer, "head": head,
                    "attention_to_first_occurrence": float(pattern[head, query_pos, first_pos].item()),
                })
        del cache
    return pd.DataFrame(rows)


def _run_activation_patching(model, clean_text: str, corrupt_text: str, layer_count: int, device: str):
    clean = model.to_tokens(clean_text).to(device)
    corrupt = model.to_tokens(corrupt_text).to(device)
    if clean.shape != corrupt.shape:
        raise ValueError(
            "Clean and corrupted prompts must tokenize to the same shape for position-aligned patching. "
            f"Got {tuple(clean.shape)} and {tuple(corrupt.shape)}. Choose a pair with equal token lengths."
        )
    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt)
    target_id = int(clean_logits[0, -1].argmax().item())
    clean_logit = float(clean_logits[0, -1, target_id].item())
    corrupt_logit = float(corrupt_logits[0, -1, target_id].item())
    rows = []
    for layer in range(layer_count):
        hook_name = f"blocks.{layer}.hook_resid_post"
        clean_source = clean_cache[hook_name].detach()
        corrupt_source = corrupt_cache[hook_name].detach()

        def patch_clean(resid, hook, source=clean_source):
            edited = resid.clone()
            edited[:, -1, :] = source[:, -1, :]
            return edited

        def patch_corrupt(resid, hook, source=corrupt_source):
            edited = resid.clone()
            edited[:, -1, :] = source[:, -1, :]
            return edited

        patched_clean = model.run_with_hooks(
            corrupt, fwd_hooks=[(hook_name, patch_clean)]
        )[0, -1, target_id]
        patched_control = model.run_with_hooks(
            corrupt, fwd_hooks=[(hook_name, patch_corrupt)]
        )[0, -1, target_id]
        patched_logit = float(patched_clean.item())
        denom = clean_logit - corrupt_logit
        rows.append({
            "layer": layer, "target_token_id": target_id,
            "clean_target_logit": clean_logit,
            "corrupted_target_logit": corrupt_logit,
            "corrupted_self_patch_control_logit": float(patched_control.item()),
            "clean_activation_patched_logit": patched_logit,
            "logit_recovery_fraction": (patched_logit - corrupt_logit) / denom if abs(denom) > 1e-12 else np.nan,
        })
    return pd.DataFrame(rows), {
        "clean_prompt": clean_text, "corrupted_prompt": corrupt_text,
        "clean_tokens": clean[0].tolist(), "corrupted_tokens": corrupt[0].tolist(),
        "target_token_id": target_id,
        "target_token": model.to_string(torch.tensor([target_id], device=device)),
        "interpretation": "Recovery is a normalized logit difference; values below 0 or above 1 are possible. The self-patch control should reproduce the corrupted baseline.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, default=12, help="Number of initial layers to evaluate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--neuron-top-k", type=int, default=3, help="Most active MLP neurons per layer to ablate; 0 disables")
    parser.add_argument("--prompt-file", type=Path, default=None, help="UTF-8 evaluation prompts, one per line")
    parser.add_argument("--discovery-prompt-file", type=Path, default=None, help="Separate UTF-8 prompt set for selecting high-activity MLP neurons")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--patch-clean", default="The capital of France is")
    parser.add_argument("--patch-corrupted", default="The capital of Germany is")
    args = parser.parse_args()
    if args.layers < 1 or args.bootstrap_samples < 100 or args.neuron_top_k < 0:
        parser.error("layers must be >= 1, bootstrap-samples >= 100, neuron-top-k >= 0")
    prompts = (
        [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.prompt_file else DEFAULT_PROMPTS
    )
    if len(prompts) < 3:
        parser.error("At least three evaluation prompts are required")
    discovery_prompts = (
        [line.strip() for line in args.discovery_prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.discovery_prompt_file else prompts
    )
    if len(discovery_prompts) < 3:
        parser.error("At least three discovery prompts are required for neuron candidate selection")
    seed_everything(args.seed)
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies first: pip install -r requirements-gpt2.txt") from exc

    device = args.device
    model = HookedTransformer.from_pretrained(args.model, device=device)
    layer_count = min(args.layers, model.cfg.n_layers)
    if layer_count != args.layers:
        print(f"Requested {args.layers} layers; model has {model.cfg.n_layers}. Using {layer_count}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running head ablations: {len(prompts)} prompts × {layer_count} layers × {model.cfg.n_heads} heads")
    head_rows, head_summary = _run_head_ablations(
        model, prompts, layer_count, args.seed, args.bootstrap_samples, device
    )
    head_rows.to_csv(args.output_dir / "gpt2_head_ablation_per_prompt.csv", index=False)
    head_summary.to_csv(args.output_dir / "gpt2_head_ablation_summary.csv", index=False)

    neuron_candidates = _find_candidate_neurons(model, discovery_prompts, layer_count, args.neuron_top_k, device)
    if neuron_candidates:
        print(f"Running MLP neuron ablations on {sum(map(len, neuron_candidates.values()))} activity-selected neurons")
    neuron_rows, neuron_summary = _run_neuron_ablations(
        model, prompts, neuron_candidates, args.seed, args.bootstrap_samples, device
    )
    if not neuron_rows.empty:
        neuron_rows.to_csv(args.output_dir / "gpt2_mlp_neuron_ablation_per_prompt.csv", index=False)
        neuron_summary.to_csv(args.output_dir / "gpt2_mlp_neuron_ablation_summary.csv", index=False)

    repeated = _run_repeated_token_probe(model, REPEATED_PROMPTS, device)
    if not repeated.empty:
        repeated.to_csv(args.output_dir / "gpt2_repeated_token_attention.csv", index=False)
        repeated.groupby(["layer", "head"], as_index=False)["attention_to_first_occurrence"].mean().to_csv(
            args.output_dir / "gpt2_repeated_token_attention_summary.csv", index=False
        )

    patch_df, patch_metadata = _run_activation_patching(
        model, args.patch_clean, args.patch_corrupted, layer_count, device
    )
    patch_df.to_csv(args.output_dir / "gpt2_activation_patching.csv", index=False)
    write_json(args.output_dir / "gpt2_activation_patching_metadata.json", patch_metadata)

    metadata = environment_metadata(args.seed, args.model, prompts)
    metadata.update({
        "layers_evaluated": layer_count,
        "heads_per_layer": model.cfg.n_heads,
        "evaluation_prompts": prompts,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "prompt-level observations within each component",
        "neuron_selection": f"top {args.neuron_top_k} mean-absolute-activation neurons per layer",
        "discovery_prompt_count": len(discovery_prompts),
        "discovery_prompt_sha256": environment_metadata(args.seed, args.model, discovery_prompts)["prompt_sha256"],
        "neuron_discovery_and_evaluation_overlap": args.discovery_prompt_file is None,
        "limitations": [
            "Ablation sensitivity is not proof of a semantic function.",
            "Prompt-level bootstrap intervals describe this prompt set, not a population of all language contexts.",
            "When no --discovery-prompt-file is supplied, neuron discovery and evaluation use the same prompts; use disjoint files for confirmatory claims.",
            "Repeated-token attention is descriptive and does not establish an induction circuit.",
        ],
    })
    write_json(args.output_dir / "gpt2_run_metadata.json", metadata)

    print("\nHighest-sensitivity attention heads (mean KL over prompts):")
    print(head_summary.head(10).to_string(index=False))
    if not neuron_summary.empty:
        print("\nHighest-sensitivity activity-selected MLP neurons (mean KL over prompts):")
        print(neuron_summary.head(10).to_string(index=False))
    print(f"\nResearch outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
