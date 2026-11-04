"""Research utilities: controlled induction probes, activation patching and statistics.

These utilities operate on the included TinyGPT. They are deliberately model-agnostic
at the tensor level and are intended as a reproducible sandbox before scaling to GPT-2.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Sequence
import math
import numpy as np
import torch


def make_induction_prompt(a: int, b: int, fillers: Sequence[int], *, device="cpu") -> torch.Tensor:
    """Create [A, B, fillers..., A]; the expected next token is B."""
    if a == b or a in fillers or b in fillers:
        raise ValueError("A, B and filler tokens must be distinct")
    return torch.tensor([[a, b, *fillers, a]], dtype=torch.long, device=device)


@torch.no_grad()
def induction_attention_score(attention: torch.Tensor, first_a: int, second_a: int) -> torch.Tensor:
    """Return attention mass from the second A query to the first A key.

    attention shape: [batch, heads, query_positions, key_positions].
    The score is descriptive evidence of an induction-like pattern, not proof of
    a circuit or causal function.
    """
    if attention.ndim != 4:
        raise ValueError("attention must have shape [batch, heads, query, key]")
    if not (0 <= first_a < attention.shape[-1] and 0 <= second_a < attention.shape[-2]):
        raise IndexError("token positions are outside the attention tensor")
    return attention[:, :, second_a, first_a].mean(0)


@torch.no_grad()
def induction_score_table(model, prompts: Iterable[torch.Tensor]):
    """Average second-occurrence-to-first-occurrence attention by layer/head.

    Each prompt must be [1, sequence] and have the form [A, B, ..., A].
    Returns a NumPy array [layers, heads].
    """
    sums = None
    count = 0
    model.eval()
    for ids in prompts:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 3:
            raise ValueError("each prompt must have shape [1, sequence] and length >= 3")
        a = int(ids[0, 0])
        positions = (ids[0] == a).nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            continue
        # Use the last occurrence as the query and the first as the matching key.
        first_a, second_a = int(positions[0]), int(positions[-1])
        model(ids)
        layer_scores = []
        for block in model.blocks:
            layer_scores.append(induction_attention_score(
                block.attn.last_attention, first_a, second_a
            ))
        scores = torch.stack(layer_scores).cpu().numpy()
        sums = scores if sums is None else sums + scores
        count += 1
    if count == 0:
        raise ValueError("no valid repeated-token prompts were supplied")
    return sums / count


@contextmanager
def patch_block_output(model, layer: int, position: int, source_activation: torch.Tensor):
    """Patch one token's block output with a clean-run activation."""
    if not (0 <= layer < len(model.blocks)):
        raise IndexError("layer out of range")
    handle_box = []

    def hook(_module, _inputs, output):
        changed = output.clone()
        changed[:, position, :] = source_activation[:, position, :].to(
            device=changed.device, dtype=changed.dtype
        )
        return changed

    handle_box.append(model.blocks[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handle_box:
            handle.remove()


@torch.no_grad()
def activation_patch_sweep(model, clean_ids: torch.Tensor, corrupted_ids: torch.Tensor,
                           target_token: int, position: int | None = None):
    """Patch each layer's residual block output from clean into corrupted input.

    Returns baseline/corrupted target logits and per-layer recovery. Inputs must
    have identical shapes and aligned token positions. `position` defaults to
    the final input position.
    """
    if clean_ids.shape != corrupted_ids.shape:
        raise ValueError("clean and corrupted inputs must have identical shapes")
    if clean_ids.ndim != 2 or clean_ids.shape[0] != 1:
        raise ValueError("currently expects batch size 1")
    position = clean_ids.shape[1] - 1 if position is None else position
    if not 0 <= position < clean_ids.shape[1]:
        raise IndexError("patch position out of range")
    model.eval()
    clean_logits, _ = model(clean_ids)
    corrupt_logits, _ = model(corrupted_ids)
    clean_target = float(clean_logits[0, -1, target_token])
    corrupt_target = float(corrupt_logits[0, -1, target_token])
    clean_cache = []
    handles = []
    try:
        for block in model.blocks:
            def save(_module, _inputs, output, bucket=clean_cache):
                bucket.append(output.detach().clone())
            handles.append(block.register_forward_hook(save))
        model(clean_ids)
    finally:
        for h in handles:
            h.remove()
    patched = []
    for layer, activation in enumerate(clean_cache):
        with patch_block_output(model, layer, position, activation):
            logits, _ = model(corrupted_ids)
        patched.append(float(logits[0, -1, target_token]))
    denom = clean_target - corrupt_target
    recovery = [((x - corrupt_target) / denom) if abs(denom) > 1e-12 else float("nan") for x in patched]
    return {
        "clean_target_logit": clean_target,
        "corrupted_target_logit": corrupt_target,
        "patched_target_logits": patched,
        "recovery_fraction": recovery,
        "target_token": int(target_token),
        "patch_position": int(position),
    }


def paired_summary(values: Sequence[float], *, bootstrap_samples: int = 2000, seed: int = 7):
    """Mean, SD, median and percentile bootstrap CI for a paired per-prompt metric."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite observations")
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples, dtype=float)
    for i in range(bootstrap_samples):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


@torch.no_grad()
def evaluate_ablation_prompts(model, prompts: Sequence[torch.Tensor], kind: str,
                              layer: int, component: int):
    """Run paired baseline-vs-ablation evaluation over a prompt collection."""
    from .core import run_ablation
    rows = []
    for i, ids in enumerate(prompts):
        result = run_ablation(model, ids, kind, layer, component)
        rows.append({
            "prompt_id": i,
            "baseline_top_id": result["baseline_top_id"],
            "ablated_top_id": result["altered_top_id"],
            "baseline_top_probability": result["baseline_top_probability"],
            "ablated_top_probability": result["altered_top_probability"],
            "kl_baseline_to_ablated": result["kl_baseline_to_altered"],
            "top_token_changed": result["baseline_top_id"] != result["altered_top_id"],
        })
    return rows


def summarize_ablation_rows(rows):
    return {
        "kl_baseline_to_ablated": paired_summary([r["kl_baseline_to_ablated"] for r in rows]),
        "top_token_change_rate": paired_summary([float(r["top_token_changed"]) for r in rows]),
        "mean_top_probability_change": paired_summary([
            r["ablated_top_probability"] - r["baseline_top_probability"] for r in rows
        ]),
    }


@torch.no_grad()
def sweep_components(model, prompts: Sequence[torch.Tensor], kind: str, layer: int,
                     components: Sequence[int] | None = None):
    """Compare every (or selected) heads/MLP neurons in one layer across prompts.

    Returns per-prompt rows and a component-level summary with prompt-level
    bootstrap intervals. This supports ranking candidates for deeper follow-up;
    ranking by KL is sensitivity analysis, not a semantic interpretation.
    """
    if kind not in {"head", "neuron"}:
        raise ValueError("kind must be 'head' or 'neuron'")
    if not 0 <= layer < len(model.blocks):
        raise IndexError("layer out of range")
    maximum = model.config.n_head if kind == "head" else model.blocks[layer].mlp.fc.out_features
    selected = list(range(maximum)) if components is None else list(components)
    if not selected or any(c < 0 or c >= maximum for c in selected):
        raise ValueError("components must be valid indices for the selected layer")
    rows = []
    from .core import run_ablation
    for component in selected:
        for prompt_id, ids in enumerate(prompts):
            result = run_ablation(model, ids, kind, layer, component)
            rows.append({
                "layer": layer, "kind": kind, "component": int(component),
                "prompt_id": prompt_id,
                "kl_baseline_to_ablated": result["kl_baseline_to_altered"],
                "top_token_changed": int(result["baseline_top_id"] != result["altered_top_id"]),
                "baseline_top_probability": result["baseline_top_probability"],
                "ablated_top_probability": result["altered_top_probability"],
            })
    summaries = []
    for component in selected:
        subset = [r for r in rows if r["component"] == component]
        stats = paired_summary([r["kl_baseline_to_ablated"] for r in subset])
        summaries.append({
            "layer": layer, "kind": kind, "component": int(component),
            **{f"kl_{k}": v for k, v in stats.items()},
            "top_token_change_rate": float(np.mean([r["top_token_changed"] for r in subset])),
        })
    summaries.sort(key=lambda r: r["kl_mean"], reverse=True)
    return rows, summaries
