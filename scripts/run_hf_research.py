#!/usr/bin/env python3
"""Run controlled Hugging Face causal-intervention experiments on a decoder-only LM.

Example (recommended model):
  python scripts/run_hf_research.py --model-id HuggingFaceTB/SmolLM3-3B-Base --device auto

The default run is intentionally bounded to a few layers/heads so it can be used as
a first reproducible analysis. Increase --layers/--max-heads/--neuron-top-k for a
larger sweep after confirming hardware capacity.
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from interpretability_lab.experiment import bootstrap_mean, environment_metadata, seed_everything, sha256_texts, write_json
from interpretability_lab.hf_backend import load_huggingface_model

DEFAULT_EVALUATION = [
    "The capital of France is",
    "Water freezes at a temperature of",
    "A triangle has three",
    "The opposite of hot is",
    "In a typical solar system, planets orbit a",
    "Python is a programming language used for",
    "The chemical symbol for oxygen is",
    "A year on Earth is approximately 365",
]
DEFAULT_DISCOVERY = [
    "The capital of Italy is",
    "A square has four",
    "The largest planet in our solar system is",
    "Plants use sunlight during",
    "An integer is a whole number that can be",
    "The boiling point of water at sea level is",
    "A function maps an input to an",
    "The Earth revolves around the",
]
DEFAULT_REPEAT = [
    " A B C D A B C",
    " red blue green yellow red blue green",
    " one two three four one two three",
    " cat dog bird fish cat dog bird",
]


def read_prompts(path: Path | None, defaults: list[str], label: str) -> list[str]:
    if path is None:
        return defaults
    if not path.exists():
        raise FileNotFoundError(f"{label} prompt file not found: {path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(prompts) < 3:
        raise ValueError(f"{label} prompt file must contain at least 3 non-empty prompts")
    return prompts


def safe_model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id).strip("_")


def attention_probe(adapter, prompts: list[str], layer_count: int) -> pd.DataFrame:
    rows: list[dict] = []
    for prompt_id, prompt in enumerate(prompts):
        ids = adapter.encode(prompt)
        if ids.shape[1] < 2:
            continue
        out = adapter.forward(ids, output_attentions=True)
        attentions = getattr(out, "attentions", None)
        if not attentions:
            raise RuntimeError(
                "Model did not return attention tensors. Ensure it was loaded with eager attention "
                "and supports output_attentions=True."
            )
        token_ids = ids[0].tolist()
        query = len(token_ids) - 1
        earlier = [i for i in range(query) if token_ids[i] == token_ids[query]]
        if not earlier:
            continue
        target = earlier[-1]
        for layer, pattern in enumerate(attentions[:layer_count]):
            # HF attention tensor is [batch, query_heads, query_position, key_position].
            final_row = pattern[0, :, query, :query].float().cpu()
            for head in range(final_row.shape[0]):
                matched_mass = float(final_row[head, target].item())
                other = torch.cat((final_row[head, :target], final_row[head, target + 1:]))
                rows.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "query_token": adapter.decode([token_ids[query]]),
                    "query_position": query,
                    "matched_previous_position": target,
                    "layer": layer,
                    "head": head,
                    "attention_to_matching_occurrence": matched_mass,
                    "mean_attention_to_other_previous_tokens": float(other.mean().item()) if other.numel() else float("nan"),
                    "match_attention_lift": matched_mass - float(other.mean().item()) if other.numel() else float("nan"),
                })
        del out
    return pd.DataFrame(rows)


def rank_neurons(adapter, prompts: list[str], layer_count: int, top_k: int) -> dict[int, list[int]]:
    totals: dict[int, torch.Tensor] = {}
    count = 0
    for prompt in prompts:
        ids = adapter.encode(prompt)
        captured = adapter.capture_mlp_intermediates(ids, range(layer_count))
        for layer, values in captured.items():
            score = values.abs().mean(dim=(0, 1))
            totals[layer] = score if layer not in totals else totals[layer] + score
        count += 1
    ranked: dict[int, list[int]] = {}
    for layer, values in totals.items():
        ranked[layer] = torch.topk(values / max(count, 1), k=min(top_k, values.numel())).indices.tolist()
    return ranked


def run_ablation_suite(adapter, prompts: list[str], layer_count: int, max_heads: int, neurons: dict[int, list[int]], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for prompt_id, prompt in enumerate(prompts):
        ids = adapter.encode(prompt)
        for layer in range(layer_count):
            for head in range(min(max_heads, adapter.n_heads)):
                row = adapter.ablation_metrics(ids, "head", layer, head)
                row.update({"prompt_id": prompt_id, "prompt": prompt})
                rows.append(row)
            for neuron in neurons.get(layer, []):
                row = adapter.ablation_metrics(ids, "neuron", layer, neuron)
                row.update({"prompt_id": prompt_id, "prompt": prompt})
                rows.append(row)
    details = pd.DataFrame(rows)
    if details.empty:
        return details, pd.DataFrame()
    group_keys = ["kind", "layer", "component"]
    summary_rows = []
    for key, group in details.groupby(group_keys, sort=True):
        stats = bootstrap_mean(group["kl_baseline_to_intervened"].tolist(), seed=seed, samples=1000)
        summary_rows.append({
            "kind": key[0], "layer": int(key[1]), "component": int(key[2]),
            **{f"kl_{name}": value for name, value in stats.items()},
            "top_token_change_rate": float(group["top_token_changed"].mean()),
            "mean_baseline_top_logit_change": float(group["baseline_top_logit_change"].mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("kl_mean", ascending=False)
    return details, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B-Base")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face branch, tag or commit hash to pin")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--layers", type=int, default=2, help="Number of initial layers to evaluate (start small)")
    parser.add_argument("--max-heads", type=int, default=4, help="Number of heads per layer to ablate")
    parser.add_argument("--neuron-top-k", type=int, default=2, help="Discovery-selected neurons per layer")
    parser.add_argument("--evaluation-prompts", type=Path, default=ROOT / "data" / "evaluation_prompts.txt")
    parser.add_argument("--discovery-prompts", type=Path, default=ROOT / "data" / "discovery_prompts.txt")
    parser.add_argument("--repeat-prompts", type=Path, default=ROOT / "data" / "repeated_token_prompts.txt")
    parser.add_argument("--patch-clean", default="The capital of France is")
    parser.add_argument("--patch-corrupted", default="The capital of Germany is")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--local-files-only", action="store_true", help="Do not access the network or download weights")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.layers < 1 or args.max_heads < 0 or args.neuron_top_k < 0:
        parser.error("--layers must be >= 1; --max-heads and --neuron-top-k must be >= 0")

    evaluation_prompts = read_prompts(args.evaluation_prompts, DEFAULT_EVALUATION, "Evaluation")
    discovery_prompts = read_prompts(args.discovery_prompts, DEFAULT_DISCOVERY, "Discovery")
    repeat_prompts = read_prompts(args.repeat_prompts, DEFAULT_REPEAT, "Repeated-token")
    seed_everything(args.seed)
    overlap = sorted(set(discovery_prompts).intersection(evaluation_prompts))
    if overlap:
        print(
            f"WARNING: discovery and evaluation sets share {len(overlap)} exact prompt(s). "
            "Treat the neuron results as exploratory or provide disjoint prompt files."
        )

    print(f"Loading {args.model_id} on {args.device} (dtype={args.dtype})...")
    adapter = load_huggingface_model(
        args.model_id, args.device, args.dtype, args.local_files_only, args.revision
    )
    layer_count = min(args.layers, adapter.n_layers)
    print(
        f"Loaded {adapter.architecture}: {adapter.n_layers} layers, {adapter.n_heads} query heads, "
        f"{adapter.n_kv_heads} KV heads, head_dim={adapter.head_dim}, "
        f"MLP width={adapter.intermediate_size}, device={adapter.device}."
    )

    run_name = f"hf_{safe_model_slug(args.model_id)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = args.output_dir / run_name
    output.mkdir(parents=True, exist_ok=True)

    print("Selecting candidate MLP neurons on discovery prompts...")
    selected_neurons = rank_neurons(adapter, discovery_prompts, layer_count, args.neuron_top_k)
    print("Running held-out prompt ablations...")
    details, summary = run_ablation_suite(adapter, evaluation_prompts, layer_count, args.max_heads, selected_neurons, args.seed)
    details.to_csv(output / "ablation_per_prompt.csv", index=False)
    summary.to_csv(output / "ablation_summary.csv", index=False)

    print("Running repeated-token attention probes...")
    repeat_df = attention_probe(adapter, repeat_prompts, layer_count)
    repeat_df.to_csv(output / "repeated_token_attention.csv", index=False)

    print("Running activation-patching sweep...")
    patch_rows: list[dict] = []
    patch_status = "completed"
    try:
        clean_ids = adapter.encode(args.patch_clean)
        corrupt_ids = adapter.encode(args.patch_corrupted)
        patch_rows = adapter.activation_patch_sweep(clean_ids, corrupt_ids, layer_count)
        patch_df = pd.DataFrame(patch_rows)
        patch_df.to_csv(output / "activation_patching.csv", index=False)
    except ValueError as exc:
        patch_status = f"skipped: {exc}"
        (output / "activation_patching_SKIPPED.txt").write_text(patch_status, encoding="utf-8")

    model_config = adapter.config.to_dict() if hasattr(adapter.config, "to_dict") else vars(adapter.config)
    metadata = environment_metadata(args.seed, args.model_id, evaluation_prompts)
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    metadata.update({
        "architecture_class": adapter.architecture,
        "requested_revision": args.revision,
        "resolved_model_commit_hash": getattr(adapter.config, "_commit_hash", None),
        "transformers_version": transformers_version,
        "model_config": model_config,
        "device": str(adapter.device),
        "dtype": str(next(adapter.model.parameters()).dtype),
        "layers_evaluated": layer_count,
        "heads_ablated_per_layer": min(args.max_heads, adapter.n_heads),
        "discovery_prompt_count": len(discovery_prompts),
        "discovery_prompt_sha256": sha256_texts(discovery_prompts),
        "discovery_evaluation_exact_overlap_count": len(overlap),
        "discovery_evaluation_exact_overlap_prompts": overlap,
        "evaluation_prompt_count": len(evaluation_prompts),
        "evaluation_prompt_sha256": sha256_texts(evaluation_prompts),
        "repeat_prompts": repeat_prompts,
        "selected_mlp_neurons": selected_neurons,
        "activation_patching_status": patch_status,
        "method_notes": [
            "Head ablation zeros one query-head context vector immediately before the attention output projection; this is GQA-aware.",
            "Neuron ablation zeros one gated MLP intermediate coordinate immediately before the down projection.",
            "MLP candidate selection uses discovery prompts; reported ablation effects use a separate evaluation prompt file.",
            "Repeated-token attention is descriptive and does not alone establish an induction circuit.",
            "Block-output activation patching replaces the final-token hidden state at one layer; equal tokenized lengths are required.",
            "Bootstrap intervals resample prompt-level observations within each component; they do not represent uncertainty over all possible prompts.",
        ],
        "limitations": [
            "Only the first configured layers and heads are evaluated by default; this is a bounded screening run.",
            "The repeated-token probe is a diagnostic, not proof of a learned circuit.",
            "Ablation can cause off-manifold activations; results should be triangulated with patching and controls.",
        ],
    })
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "selected_neurons.json", selected_neurons)

    report_lines = [
        f"# Hugging Face mechanistic interpretability run: {args.model_id}",
        "",
        f"- UTC timestamp: {metadata['created_utc']}",
        f"- Layers evaluated: {layer_count}/{adapter.n_layers}",
        f"- Evaluation prompts: {len(evaluation_prompts)} (SHA-256: `{metadata['evaluation_prompt_sha256']}`)",
        f"- Discovery prompts: {len(discovery_prompts)} (SHA-256: `{metadata['discovery_prompt_sha256']}`)",
        f"- Device/dtype: {metadata['device']} / {metadata['dtype']}",
        "",
        "## Highest measured intervention effects",
        "",
    ]
    if not summary.empty:
        report_lines.append(summary.head(10).to_markdown(index=False))
    else:
        report_lines.append("No ablation rows were produced.")
    report_lines += [
        "",
        "## Interpretation cautions",
        "",
        "These are measured effects on the supplied prompt set, not universal functional labels. "
        "Inspect per-prompt CSVs, uncertainty intervals, discovery/evaluation separation, and controls "
        "before claiming a component has a particular function.",
        "",
        f"Activation patching: {patch_status}",
    ]
    (output / "RUN_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nFinished. Results saved to: {output}")
    print("Open the Hugging Face dashboard and select this run to explore the results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
