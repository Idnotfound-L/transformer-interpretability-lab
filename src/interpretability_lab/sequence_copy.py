"""Controlled sequence-copying studies for decoder-only Hugging Face models.

Task: [prefix..., A, B, filler..., A] -> predict B.
All prompts have equal length and discovery/evaluation token pools are disjoint.
Attention scores are descriptive; causal claims are based on held-out interventions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class SequenceCopyCase:
    prompt_id: int
    input_ids: tuple[int, ...]
    expected_token_id: int
    repeated_token_id: int
    first_occurrence_position: int
    successor_position: int
    query_position: int
    corruption_token_id: int


def lexical_token_ids(tokenizer: Any) -> list[int]:
    """Return ordinary alphabetic single-token IDs, excluding special tokens."""
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    result: list[int] = []
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    word_start: list[int] = []
    fallback: list[int] = []
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        try:
            raw = convert(token_id) if callable(convert) else tokenizer.decode([token_id])
            if isinstance(raw, list):
                raw = raw[0]
            raw_text = str(raw)
            token = raw_text.replace("Ġ", "").replace("▁", "").strip()
            if 2 <= len(token) <= 14 and token.isalpha():
                fallback.append(token_id)
                if raw_text.startswith(("Ġ", "▁")):
                    word_start.append(token_id)
        except Exception:
            continue
    # Prefer tokens that begin a word under common BPE/SentencePiece schemes.
    # Fall back for tokenizers that do not expose explicit word-boundary markers.
    result = word_start if len(word_start) >= 32 else fallback
    if len(result) < 32:
        raise RuntimeError(
            "Could not find enough alphabetic single-token vocabulary entries. "
            "Use a standard decoder-only tokenizer with a conventional vocabulary."
        )
    return result


def split_disjoint_token_pools(token_ids: Sequence[int], seed: int = 7) -> tuple[list[int], list[int]]:
    """Split token identities into non-overlapping discovery and evaluation pools."""
    if len(set(token_ids)) != len(token_ids):
        token_ids = list(dict.fromkeys(token_ids))
    if len(token_ids) < 32:
        raise ValueError("At least 32 distinct token IDs are required")
    shuffled = np.random.default_rng(seed).permutation(np.asarray(token_ids, dtype=np.int64)).tolist()
    midpoint = len(shuffled) // 2
    return [int(x) for x in shuffled[:midpoint]], [int(x) for x in shuffled[midpoint:]]


def make_sequence_copy_cases(
    token_pool: Sequence[int], count: int, seed: int, *, start_prompt_id: int = 0
) -> list[SequenceCopyCase]:
    """Create fixed-length prompts with randomized repeated-token locations.

    Prompt layout (length 9): [prefix(1..3), A, B, fillers, A].
    B is the expected next token after the final A. Each prompt uses eight
    distinct sequence-token IDs plus a distinct corruption token; tokens are
    allocated without replacement across prompts within this split.
    """
    pool = list(dict.fromkeys(int(x) for x in token_pool))
    if len(pool) < 10:
        raise ValueError("Token pool must contain at least 10 distinct IDs")
    if count < 1:
        raise ValueError("count must be positive")
    required = count * 9
    if len(pool) < required:
        raise ValueError(
            f"Need at least {required} distinct token IDs for {count} prompts; "
            f"the supplied split contains {len(pool)}. Reduce case counts or use a tokenizer "
            "with a larger alphabetic-token pool."
        )
    rng = np.random.default_rng(seed)
    # Allocate each token ID to exactly one prompt in this split. This avoids
    # prompt-level bootstrap samples sharing token identities by construction.
    allocated = [int(x) for x in rng.choice(pool, size=required, replace=False)]
    cases: list[SequenceCopyCase] = []
    for i in range(count):
        chosen = allocated[i * 9:(i + 1) * 9]
        prefix_len = int(rng.integers(1, 4))
        prefix = chosen[:prefix_len]
        a_token = chosen[prefix_len]
        b_token = chosen[prefix_len + 1]
        fillers = chosen[prefix_len + 2:8]
        corruption = chosen[8]
        ids = tuple(prefix + [a_token, b_token] + fillers + [a_token])
        if len(ids) != 9 or len(set(ids[:-1])) != 8 or ids[-1] != a_token:
            raise AssertionError("Internal sequence-copy task construction error")
        a_pos = prefix_len
        cases.append(SequenceCopyCase(
            prompt_id=start_prompt_id + i,
            input_ids=ids,
            expected_token_id=b_token,
            repeated_token_id=a_token,
            first_occurrence_position=a_pos,
            successor_position=a_pos + 1,
            query_position=len(ids) - 1,
            corruption_token_id=corruption,
        ))
    return cases


def choose_layer_indices(total_layers: int, requested_layers: int) -> list[int]:
    """Choose evenly spaced layers across the full model depth."""
    if total_layers < 1 or requested_layers < 1:
        raise ValueError("total_layers and requested_layers must be positive")
    if requested_layers >= total_layers:
        return list(range(total_layers))
    if requested_layers == 1:
        return [total_layers // 2]
    return sorted(set(int(round(x)) for x in np.linspace(0, total_layers - 1, requested_layers)))


def batch_input_ids(cases: Sequence[SequenceCopyCase], device: torch.device | str) -> torch.Tensor:
    return torch.tensor([case.input_ids for case in cases], dtype=torch.long, device=device)


def collect_copy_attention_scores(
    adapter: Any,
    cases: Sequence[SequenceCopyCase],
    layers: Sequence[int],
    batch_size: int = 16,
) -> pd.DataFrame:
    """Measure final-query attention to the earlier A occurrence for each head."""
    rows: list[dict[str, Any]] = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        ids = batch_input_ids(chunk, adapter.device)
        out = adapter.forward(ids, output_attentions=True)
        attentions = getattr(out, "attentions", None)
        if attentions is None:
            raise RuntimeError(
                "Model returned no attention tensors. Load it with eager attention and verify "
                "that this Transformers architecture supports output_attentions=True."
            )
        for layer in layers:
            pattern = attentions[layer].float()
            for batch_index, case in enumerate(chunk):
                row = pattern[batch_index, :, case.query_position, :case.query_position]
                target = row[:, case.first_occurrence_position]
                if row.shape[-1] > 1:
                    other = torch.cat((
                        row[:, :case.first_occurrence_position],
                        row[:, case.first_occurrence_position + 1:]
                    ), dim=-1)
                    other_mean = other.mean(-1)
                else:
                    other_mean = torch.zeros_like(target)
                for head in range(row.shape[0]):
                    mass = float(target[head].item())
                    other_value = float(other_mean[head].item())
                    rows.append({
                        "prompt_id": case.prompt_id,
                        "layer": int(layer),
                        "head": int(head),
                        "first_A_position": case.first_occurrence_position,
                        "query_position": case.query_position,
                        "attention_to_first_A": mass,
                        "mean_attention_to_other_previous_tokens": other_value,
                        "attention_lift": mass - other_value,
                    })
        del out
    return pd.DataFrame(rows)


def rank_mlp_neurons(
    adapter: Any,
    cases: Sequence[SequenceCopyCase],
    layers: Sequence[int],
    batch_size: int = 8,
) -> dict[int, list[tuple[int, float]]]:
    """Rank post-gating MLP coordinates by mean absolute final-position activity."""
    totals: dict[int, torch.Tensor] = {}
    n = 0
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        ids = batch_input_ids(chunk, adapter.device)
        captured = adapter.capture_mlp_intermediates(ids, layers)
        for layer, activations in captured.items():
            values = activations[:, -1, :].abs().mean(dim=0).float().cpu()
            totals[layer] = values if layer not in totals else totals[layer] + values
        n += len(chunk)
    if not n:
        raise ValueError("Cannot rank neurons without discovery cases")
    return {
        int(layer): [(int(i), float(values[i].item())) for i in torch.argsort(values / n, descending=True).tolist()]
        for layer, values in totals.items()
    }


def build_component_candidates(
    attention_scores: pd.DataFrame,
    neuron_scores: dict[int, list[tuple[int, float]]],
    layers: Sequence[int],
    n_heads: int,
    intermediate_size: int,
    top_per_layer: int,
    seed: int,
) -> pd.DataFrame:
    """Select discovery-ranked candidates and matched random controls per layer."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for layer in layers:
        heads = (attention_scores[attention_scores.layer == layer]
                 .groupby("head")["attention_lift"].mean().sort_values(ascending=False))
        selected_heads = [int(x) for x in heads.head(top_per_layer).index.tolist()]
        available_heads = [h for h in range(n_heads) if h not in selected_heads]
        for rank, head in enumerate(selected_heads):
            rows.append({"kind": "head", "layer": int(layer), "component": head,
                         "selection": "discovery_top", "rank": rank + 1,
                         "discovery_score": float(heads.loc[head]),
                         "pair_id": f"head_L{layer}_R{rank + 1}"})
            if available_heads:
                control = int(rng.choice(available_heads))
                available_heads.remove(control)
                rows.append({"kind": "head", "layer": int(layer), "component": control,
                             "selection": "random_control", "rank": rank + 1,
                             "discovery_score": float("nan"),
                             "pair_id": f"head_L{layer}_R{rank + 1}"})
        ranked_neurons = neuron_scores.get(layer, [])
        selected_neurons = ranked_neurons[:top_per_layer]
        available_neurons = set(range(intermediate_size)) - {int(x[0]) for x in selected_neurons}
        for rank, (neuron, score) in enumerate(selected_neurons):
            rows.append({"kind": "neuron", "layer": int(layer), "component": int(neuron),
                         "selection": "discovery_top", "rank": rank + 1,
                         "discovery_score": float(score),
                         "pair_id": f"neuron_L{layer}_R{rank + 1}"})
            if available_neurons:
                control = int(rng.choice(np.fromiter(available_neurons, dtype=np.int64)))
                available_neurons.remove(control)
                rows.append({"kind": "neuron", "layer": int(layer), "component": control,
                             "selection": "random_control", "rank": rank + 1,
                             "discovery_score": float("nan"),
                             "pair_id": f"neuron_L{layer}_R{rank + 1}"})
    return pd.DataFrame(rows)


def _prediction_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, targets[:, None], float("-inf"))
    margin = target_logits - other.max(dim=1).values
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    target_prob = probs.gather(1, targets[:, None]).squeeze(1)
    return {
        "target_logit": target_logits,
        "margin": margin,
        "target_probability": target_prob,
        "correct": logits.argmax(dim=-1).eq(targets),
        "log_probs": log_probs,
    }


def evaluate_copy_interventions(
    adapter: Any,
    cases: Sequence[SequenceCopyCase],
    candidates: pd.DataFrame,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate final-query-only interventions on held-out prompts."""
    cached: list[tuple[list[SequenceCopyCase], torch.Tensor, torch.Tensor]] = []
    baseline_rows: list[dict[str, Any]] = []
    for start in range(0, len(cases), batch_size):
        chunk = list(cases[start:start + batch_size])
        ids = batch_input_ids(chunk, adapter.device)
        logits = adapter.forward(ids).logits[:, -1, :].float()
        targets = torch.tensor([c.expected_token_id for c in chunk], device=adapter.device)
        metrics = _prediction_metrics(logits, targets)
        cached.append((chunk, ids, logits.detach()))
        for i, case in enumerate(chunk):
            baseline_rows.append({
                "prompt_id": case.prompt_id,
                "baseline_correct": bool(metrics["correct"][i].item()),
                "baseline_target_probability": float(metrics["target_probability"][i].item()),
                "baseline_target_logit": float(metrics["target_logit"][i].item()),
                "baseline_copy_margin": float(metrics["margin"][i].item()),
            })

    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        kind = str(candidate.kind)
        layer = int(candidate.layer)
        component = int(candidate.component)
        cm = (adapter.ablate_head(layer, component, position=-1) if kind == "head"
              else adapter.ablate_neuron(layer, component, position=-1))
        with cm:
            for chunk, ids, base_logits in cached:
                altered_logits = adapter.forward(ids).logits[:, -1, :].float()
                targets = torch.tensor([c.expected_token_id for c in chunk], device=adapter.device)
                base = _prediction_metrics(base_logits, targets)
                changed = _prediction_metrics(altered_logits, targets)
                kl = (base["log_probs"].exp() * (base["log_probs"] - changed["log_probs"])).sum(-1)
                for i, case in enumerate(chunk):
                    rows.append({
                        "prompt_id": case.prompt_id,
                        "kind": kind,
                        "layer": layer,
                        "component": component,
                        "selection": str(candidate.selection),
                        "pair_id": str(candidate.pair_id),
                        "expected_token_id": case.expected_token_id,
                        "baseline_correct": bool(base["correct"][i].item()),
                        "intervened_correct": bool(changed["correct"][i].item()),
                        "baseline_copy_margin": float(base["margin"][i].item()),
                        "intervened_copy_margin": float(changed["margin"][i].item()),
                        "copy_margin_drop": float((base["margin"][i] - changed["margin"][i]).item()),
                        "baseline_target_probability": float(base["target_probability"][i].item()),
                        "intervened_target_probability": float(changed["target_probability"][i].item()),
                        "target_probability_change": float((changed["target_probability"][i] - base["target_probability"][i]).item()),
                        "target_logit_change": float((changed["target_logit"][i] - base["target_logit"][i]).item()),
                        "kl_baseline_to_intervened": float(kl[i].item()),
                    })
    return pd.DataFrame(baseline_rows), pd.DataFrame(rows)


def _bootstrap_ci(values: Sequence[float], seed: int, samples: int = 5000) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(samples, x.size), replace=True).mean(axis=1)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _permutation_signflip_p(values: Sequence[float], seed: int, samples: int = 10000) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    observed = abs(float(x.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, x.size))
    null = np.abs((signs * x).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (samples + 1))


def _benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rev_index in range(m - 1, -1, -1):
        rank = rev_index + 1
        running = min(running, p[order[rev_index]] * m / rank)
        adjusted[rev_index] = running
    q[order] = np.clip(adjusted, 0, 1)
    return q.tolist()


def summarize_interventions(rows: pd.DataFrame, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize component effects and paired candidate-vs-random-control contrasts."""
    summary_rows: list[dict[str, Any]] = []
    keys = ["kind", "layer", "component", "selection", "pair_id"]
    for key, group in rows.groupby(keys, sort=True):
        stats = _bootstrap_ci(group.copy_margin_drop.tolist(), seed + int(key[1]) * 101 + int(key[2]))
        summary_rows.append({
            "kind": key[0], "layer": int(key[1]), "component": int(key[2]),
            "selection": key[3], "pair_id": key[4],
            **{f"margin_drop_{k}": v for k, v in stats.items()},
            "mean_kl": float(group.kl_baseline_to_intervened.mean()),
            "copy_accuracy_after_ablation": float(group.intervened_correct.mean()),
            "copy_accuracy_before_ablation": float(group.baseline_correct.mean()),
            "top1_flip_rate": float((group.baseline_correct != group.intervened_correct).mean()),
        })
    summary = pd.DataFrame(summary_rows)

    paired_rows: list[dict[str, Any]] = []
    for (kind, layer, pair_id), group in rows.groupby(["kind", "layer", "pair_id"], sort=True):
        pivot = group.pivot(index="prompt_id", columns="selection", values="copy_margin_drop")
        if not {"discovery_top", "random_control"}.issubset(set(pivot.columns)):
            continue
        paired = (pivot["discovery_top"] - pivot["random_control"]).dropna().to_numpy()
        if len(paired) == 0:
            continue
        stats = _bootstrap_ci(paired, seed + int(layer) * 313)
        p_value = _permutation_signflip_p(paired, seed + int(layer) * 997)
        paired_rows.append({
            "kind": kind, "layer": int(layer), "pair_id": pair_id,
            "n_prompts": int(len(paired)),
            "selected_minus_random_mean_margin_drop": stats["mean"],
            "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            "permutation_p": p_value,
        })
    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        paired_df["q_bh"] = _benjamini_hochberg(paired_df.permutation_p.tolist())
    return summary, paired_df



def summarize_baseline_metrics(baseline: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Bootstrap uncertainty for held-out copy accuracy and target margins."""
    metrics = {
        "copy_accuracy": baseline.baseline_correct.astype(float).tolist(),
        "target_probability": baseline.baseline_target_probability.astype(float).tolist(),
        "copy_margin": baseline.baseline_copy_margin.astype(float).tolist(),
    }
    rows = []
    for offset, (name, values) in enumerate(metrics.items()):
        stats = _bootstrap_ci(values, seed + offset)
        rows.append({"metric": name, **stats})
    return pd.DataFrame(rows)


def summarize_activation_patching(patching: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Bootstrap intervals for patching recovery by layer and patch location."""
    rows = []
    for (layer, location), group in patching.groupby(["layer", "patch_location"], sort=True):
        margin_stats = _bootstrap_ci(group.margin_recovery.tolist(), seed + int(layer) * 17)
        logit_stats = _bootstrap_ci(group.target_logit_recovery.tolist(), seed + int(layer) * 29 + 1)
        rows.append({
            "layer": int(layer),
            "patch_location": location,
            "n": int(margin_stats["n"]),
            "mean_margin_recovery": margin_stats["mean"],
            "margin_recovery_ci95_low": margin_stats["ci95_low"],
            "margin_recovery_ci95_high": margin_stats["ci95_high"],
            "mean_target_logit_recovery": logit_stats["mean"],
            "target_logit_recovery_ci95_low": logit_stats["ci95_low"],
            "target_logit_recovery_ci95_high": logit_stats["ci95_high"],
        })
    return pd.DataFrame(rows)

def _capture_block_outputs(adapter: Any, input_ids: torch.Tensor, layers: Sequence[int]) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        block = adapter.layers[int(layer)].block
        def hook(_module, _inputs, output, key=int(layer)):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[key] = hidden.detach()
        handles.append(block.register_forward_hook(hook))
    try:
        adapter.forward(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def run_copy_activation_patching(
    adapter: Any,
    cases: Sequence[SequenceCopyCase],
    layers: Sequence[int],
) -> pd.DataFrame:
    """Patch clean successor-token or query-position block states into corrupted runs."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        clean_ids = batch_input_ids([case], adapter.device)
        corrupted = list(case.input_ids)
        corrupted[case.successor_position] = case.corruption_token_id
        corrupted_ids = torch.tensor([corrupted], dtype=torch.long, device=adapter.device)
        clean_logits = adapter.forward(clean_ids).logits[0, -1].float()
        corrupt_logits = adapter.forward(corrupted_ids).logits[0, -1].float()
        target = case.expected_token_id
        clean_margin = float((clean_logits[target] - torch.cat((clean_logits[:target], clean_logits[target + 1:])).max()).item())
        corrupt_margin = float((corrupt_logits[target] - torch.cat((corrupt_logits[:target], corrupt_logits[target + 1:])).max()).item())
        clean_acts = _capture_block_outputs(adapter, clean_ids, layers)
        for layer in layers:
            for location, position in (("corrupted_successor", case.successor_position),
                                       ("final_query", case.query_position)):
                source = clean_acts[int(layer)]
                def patch(_module, _inputs, output, src=source, pos=position):
                    is_tuple = isinstance(output, tuple)
                    hidden = output[0] if is_tuple else output
                    hidden = hidden.clone()
                    hidden[:, pos, :] = src[:, pos, :].to(device=hidden.device, dtype=hidden.dtype)
                    return (hidden, *output[1:]) if is_tuple else hidden
                handle = adapter.layers[int(layer)].block.register_forward_hook(patch)
                try:
                    patched = adapter.forward(corrupted_ids).logits[0, -1].float()
                finally:
                    handle.remove()
                patched_margin = float((patched[target] - torch.cat((patched[:target], patched[target + 1:])).max()).item())
                clean_logit = float(clean_logits[target].item())
                corrupt_logit = float(corrupt_logits[target].item())
                patched_logit = float(patched[target].item())
                logit_denominator = clean_logit - corrupt_logit
                margin_denominator = clean_margin - corrupt_margin
                rows.append({
                    "prompt_id": case.prompt_id,
                    "layer": int(layer),
                    "patch_location": location,
                    "patched_position": int(position),
                    "target_token_id": int(target),
                    "clean_target_logit": clean_logit,
                    "corrupted_target_logit": corrupt_logit,
                    "patched_target_logit": patched_logit,
                    "target_logit_recovery": ((patched_logit - corrupt_logit) / logit_denominator
                                               if abs(logit_denominator) > 1e-10 else float("nan")),
                    "clean_copy_margin": clean_margin,
                    "corrupted_copy_margin": corrupt_margin,
                    "patched_copy_margin": patched_margin,
                    "margin_recovery": ((patched_margin - corrupt_margin) / margin_denominator
                                        if abs(margin_denominator) > 1e-10 else float("nan")),
                })
    return pd.DataFrame(rows)
