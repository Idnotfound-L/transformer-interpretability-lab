"""Protocol V2: task-specific attention/MLP discovery and non-trivial causal patching.

Primary task: [prefix..., A, B, fillers..., A] -> predict B.
Primary induction-style attention target: the earlier successor B position.
Control task: random unique-token sequence -> independently sampled random next token.
All token identity pools are disjoint across discovery, copy evaluation, and control evaluation.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .sequence_copy import (
    SequenceCopyCase,
    batch_input_ids,
    _prediction_metrics,
    _bootstrap_ci,
    _permutation_signflip_p,
    _benjamini_hochberg,
)


def split_three_disjoint_token_pools(token_ids: Sequence[int], seed: int = 7):
    unique = list(dict.fromkeys(int(x) for x in token_ids))
    if len(unique) < 90:
        # Case-generation functions perform exact per-split allocation checks.
        raise ValueError("At least 90 distinct token IDs are required")
    shuffled = np.random.default_rng(seed).permutation(np.asarray(unique, dtype=np.int64)).tolist()
    third = len(shuffled) // 3
    return (
        [int(x) for x in shuffled[:third]],
        [int(x) for x in shuffled[third:2 * third]],
        [int(x) for x in shuffled[2 * third:]],
    )


def make_random_transition_control_cases(
    token_pool: Sequence[int], count: int, seed: int, *, start_prompt_id: int = 200000
) -> list[SequenceCopyCase]:
    """Build length-matched sequences with no repeated token and an independent target.

    Each case consumes 11 distinct IDs: 9 prompt IDs, one target ID, one corruption ID.
    This is a generic next-token sensitivity control, not a language modelling benchmark.
    """
    pool = list(dict.fromkeys(int(x) for x in token_pool))
    required = count * 11
    if len(pool) < required:
        raise ValueError(f"Need {required} distinct IDs for {count} control cases; got {len(pool)}")
    rng = np.random.default_rng(seed)
    allocation = [int(x) for x in rng.choice(pool, size=required, replace=False)]
    cases: list[SequenceCopyCase] = []
    for i in range(count):
        chunk = allocation[i * 11:(i + 1) * 11]
        ids = tuple(chunk[:9])
        target, corruption = chunk[9], chunk[10]
        cases.append(SequenceCopyCase(
            prompt_id=start_prompt_id + i,
            input_ids=ids,
            expected_token_id=target,
            repeated_token_id=ids[-1],
            first_occurrence_position=0,
            successor_position=1,
            query_position=8,
            corruption_token_id=corruption,
        ))
    return cases


def collect_copy_attention_scores_v2(
    adapter: Any, cases: Sequence[SequenceCopyCase], layers: Sequence[int], batch_size: int = 16
) -> pd.DataFrame:
    """Score attention to the earlier successor B; retain earlier A as a diagnostic."""
    rows: list[dict[str, Any]] = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        ids = batch_input_ids(chunk, adapter.device)
        out = adapter.forward(ids, output_attentions=True)
        attentions = getattr(out, "attentions", None)
        if attentions is None:
            raise RuntimeError("No attention tensors returned; use eager attention implementation")
        for layer in layers:
            pattern = attentions[int(layer)].float()
            for bi, case in enumerate(chunk):
                # Exclude the query's self-position: only earlier keys are considered.
                row = pattern[bi, :, case.query_position, :case.query_position]
                b_pos, a_pos = case.successor_position, case.first_occurrence_position
                b_mass = row[:, b_pos]
                a_mass = row[:, a_pos]
                other_b = torch.cat((row[:, :b_pos], row[:, b_pos + 1:]), dim=-1)
                other_a = torch.cat((row[:, :a_pos], row[:, a_pos + 1:]), dim=-1)
                b_other = other_b.mean(-1) if other_b.shape[-1] else torch.zeros_like(b_mass)
                a_other = other_a.mean(-1) if other_a.shape[-1] else torch.zeros_like(a_mass)
                for head in range(row.shape[0]):
                    rows.append({
                        "prompt_id": case.prompt_id,
                        "layer": int(layer),
                        "head": int(head),
                        "first_A_position": int(a_pos),
                        "successor_B_position": int(b_pos),
                        "query_position": int(case.query_position),
                        "attention_to_first_A": float(a_mass[head].item()),
                        "attention_lift_to_first_A": float((a_mass[head] - a_other[head]).item()),
                        "attention_to_successor_B": float(b_mass[head].item()),
                        "attention_lift_to_successor_B": float((b_mass[head] - b_other[head]).item()),
                    })
        del out
    return pd.DataFrame(rows)


def rank_mlp_neurons_v2(
    adapter: Any, cases: Sequence[SequenceCopyCase], layers: Sequence[int], batch_size: int = 8
) -> dict[int, list[tuple[int, float, float]]]:
    """Rank MLP coordinates by final-query activation change when B is corrupted.

    Tuple fields are (neuron_index, mean_abs_clean_corrupt_delta, mean_abs_activation).
    The second quantity is the discovery score; the third supports activity-matched controls.
    """
    delta_totals: dict[int, torch.Tensor] = {}
    activity_totals: dict[int, torch.Tensor] = {}
    n = 0
    for start in range(0, len(cases), batch_size):
        chunk = list(cases[start:start + batch_size])
        clean_ids = batch_input_ids(chunk, adapter.device)
        corrupt_rows = [list(c.input_ids) for c in chunk]
        for row, case in zip(corrupt_rows, chunk):
            row[case.successor_position] = case.corruption_token_id
        corrupt_ids = torch.tensor(corrupt_rows, dtype=torch.long, device=adapter.device)
        clean_acts = adapter.capture_mlp_intermediates(clean_ids, layers)
        corrupt_acts = adapter.capture_mlp_intermediates(corrupt_ids, layers)
        for layer in layers:
            clean = clean_acts[int(layer)][:, -1, :].float()
            corrupt = corrupt_acts[int(layer)][:, -1, :].float()
            delta = (clean - corrupt).abs().mean(dim=0)
            activity = (clean.abs().mean(dim=0) + corrupt.abs().mean(dim=0)) / 2.0
            delta_totals[int(layer)] = delta * len(chunk) if int(layer) not in delta_totals else delta_totals[int(layer)] + delta * len(chunk)
            activity_totals[int(layer)] = activity * len(chunk) if int(layer) not in activity_totals else activity_totals[int(layer)] + activity * len(chunk)
        n += len(chunk)
    if n == 0:
        raise ValueError("Cannot rank neurons without discovery cases")
    result = {}
    for layer in layers:
        delta = delta_totals[int(layer)] / n
        activity = activity_totals[int(layer)] / n
        order = torch.argsort(delta, descending=True).tolist()
        result[int(layer)] = [
            (int(i), float(delta[i].item()), float(activity[i].item())) for i in order
        ]
    return result


def build_component_candidates_v2(
    attention_scores: pd.DataFrame,
    neuron_scores: dict[int, list[tuple[int, float, float]]],
    layers: Sequence[int], n_heads: int, intermediate_size: int,
    top_per_layer: int, seed: int,
) -> pd.DataFrame:
    """Select B-attending heads and task-sensitive neurons with matched controls."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    head_score_col = "attention_lift_to_successor_B"
    if head_score_col not in attention_scores.columns:
        raise ValueError(f"V2 attention scores must include {head_score_col}")
    for layer in layers:
        head_scores = (attention_scores[attention_scores.layer == layer]
                       .groupby("head")[head_score_col].mean().sort_values(ascending=False))
        selected_heads = [int(x) for x in head_scores.head(top_per_layer).index.tolist()]
        available_heads = [h for h in range(n_heads) if h not in selected_heads]
        for rank, head in enumerate(selected_heads):
            pair_id = f"head_L{layer}_R{rank + 1}"
            rows.append({"kind": "head", "layer": int(layer), "component": head,
                         "selection": "discovery_top", "rank": rank + 1,
                         "discovery_score": float(head_scores.loc[head]),
                         "mean_abs_activity": float("nan"), "pair_id": pair_id})
            if available_heads:
                control = int(rng.choice(available_heads))
                available_heads.remove(control)
                rows.append({"kind": "head", "layer": int(layer), "component": control,
                             "selection": "random_control", "rank": rank + 1,
                             "discovery_score": float("nan"), "mean_abs_activity": float("nan"),
                             "pair_id": pair_id})

        neurons = neuron_scores.get(int(layer), [])
        selected_neurons = neurons[:top_per_layer]
        all_ids = {int(item[0]) for item in neurons}
        for rank, selected in enumerate(selected_neurons):
            neuron, score, activity = int(selected[0]), float(selected[1]), float(selected[2])
            pair_id = f"neuron_L{layer}_R{rank + 1}"
            rows.append({"kind": "neuron", "layer": int(layer), "component": neuron,
                         "selection": "discovery_top", "rank": rank + 1,
                         "discovery_score": score, "mean_abs_activity": activity,
                         "pair_id": pair_id})
            alternatives = [item for item in neurons if int(item[0]) != neuron]
            if alternatives:
                # Match control to similar baseline activity magnitude, then sample
                # from the 10 closest available coordinates to avoid picking the same unit.
                alternatives.sort(key=lambda item: abs(float(item[2]) - activity))
                pool = alternatives[:min(10, len(alternatives))]
                control_item = pool[int(rng.integers(0, len(pool)))]
                control = int(control_item[0])
                rows.append({"kind": "neuron", "layer": int(layer), "component": control,
                             "selection": "random_control", "control_matching": "activity_matched", "rank": rank + 1,
                             "discovery_score": float(control_item[1]),
                             "mean_abs_activity": float(control_item[2]), "pair_id": pair_id})
    frame = pd.DataFrame(rows)
    if "control_matching" not in frame.columns:
        frame["control_matching"] = "random_component"
    else:
        frame["control_matching"] = frame["control_matching"].fillna("random_component")
    return frame


def _capture_v2_activations(adapter: Any, input_ids: torch.Tensor, layers: Sequence[int]):
    block_out: dict[int, torch.Tensor] = {}
    head_in: dict[int, torch.Tensor] = {}
    mlp_in: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        item = adapter.layers[int(layer)]
        def block_hook(_m, _i, out, key=int(layer)):
            value = out[0] if isinstance(out, tuple) else out
            block_out[key] = value.detach()
        def head_hook(_m, inputs, key=int(layer)):
            head_in[key] = inputs[0].detach()
        def mlp_hook(_m, inputs, key=int(layer)):
            mlp_in[key] = inputs[0].detach()
        handles.extend([
            item.block.register_forward_hook(block_hook),
            item.attention_output.register_forward_pre_hook(head_hook),
            item.mlp_output.register_forward_pre_hook(mlp_hook),
        ])
    try:
        adapter.forward(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return block_out, head_in, mlp_in


def _logit_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, targets[:, None], float("-inf"))
    return target_logits - masked.max(dim=-1).values


def run_copy_activation_patching_v2(
    adapter: Any, cases: Sequence[SequenceCopyCase], layers: Sequence[int],
    candidates: pd.DataFrame, seed: int = 7,
) -> pd.DataFrame:
    """Run non-trivial block, head-specific and neuron-specific patching plus controls.

    No final-query whole-block patch is performed. Component patches restore only one
    selected head/neuron's final-query contribution from the clean run into a corrupted run.
    """
    if not cases:
        return pd.DataFrame()
    clean_ids = batch_input_ids(cases, adapter.device)
    corrupt_rows = [list(c.input_ids) for c in cases]
    for row, case in zip(corrupt_rows, cases):
        row[case.successor_position] = case.corruption_token_id
    corrupt_ids = torch.tensor(corrupt_rows, dtype=torch.long, device=adapter.device)
    targets = torch.tensor([c.expected_token_id for c in cases], dtype=torch.long, device=adapter.device)
    query_positions = torch.tensor([c.query_position for c in cases], dtype=torch.long, device=adapter.device)
    successor_positions = torch.tensor([c.successor_position for c in cases], dtype=torch.long, device=adapter.device)

    with torch.inference_mode():
        clean_logits = adapter.forward(clean_ids).logits[:, -1, :].float()
        corrupt_logits = adapter.forward(corrupt_ids).logits[:, -1, :].float()
        clean_margin = _logit_margin(clean_logits, targets)
        corrupt_margin = _logit_margin(corrupt_logits, targets)
        clean_target_logit = clean_logits.gather(1, targets[:, None]).squeeze(1)
        corrupt_target_logit = corrupt_logits.gather(1, targets[:, None]).squeeze(1)
        block_clean, head_clean, mlp_clean = _capture_v2_activations(adapter, clean_ids, layers)

        rng = np.random.default_rng(seed)
        random_positions = []
        for case in cases:
            eligible = [p for p in range(case.query_position)
                        if p not in {case.first_occurrence_position, case.successor_position}]
            if not eligible:
                eligible = [p for p in range(case.query_position) if p != case.successor_position]
            random_positions.append(int(rng.choice(eligible)))
        random_positions_t = torch.tensor(random_positions, dtype=torch.long, device=adapter.device)
        batch_idx = torch.arange(len(cases), device=adapter.device)
        rows: list[dict[str, Any]] = []

        def record(layer, location, selection, kind, component, positions, patched_logits):
            patched_logits = patched_logits[:, -1, :].float() if patched_logits.ndim == 3 else patched_logits.float()
            patched_margin = _logit_margin(patched_logits, targets)
            patched_target_logit = patched_logits.gather(1, targets[:, None]).squeeze(1)
            margin_den = clean_margin - corrupt_margin
            logit_den = clean_target_logit - corrupt_target_logit
            for i, case in enumerate(cases):
                rows.append({
                    "prompt_id": int(case.prompt_id), "layer": int(layer),
                    "patch_location": location, "kind": kind, "selection": selection,
                    "component": component, "patched_position": int(positions[i]),
                    "clean_copy_margin": float(clean_margin[i].item()),
                    "corrupted_copy_margin": float(corrupt_margin[i].item()),
                    "patched_copy_margin": float(patched_margin[i].item()),
                    "margin_recovery": float(((patched_margin[i] - corrupt_margin[i]) / margin_den[i]).item())
                        if abs(float(margin_den[i].item())) > 1e-10 else float("nan"),
                    "clean_target_logit": float(clean_target_logit[i].item()),
                    "corrupted_target_logit": float(corrupt_target_logit[i].item()),
                    "patched_target_logit": float(patched_target_logit[i].item()),
                    "target_logit_recovery": float(((patched_target_logit[i] - corrupt_target_logit[i]) / logit_den[i]).item())
                        if abs(float(logit_den[i].item())) > 1e-10 else float("nan"),
                })

        for layer in layers:
            item = adapter.layers[int(layer)]
            # Whole-block state patch at the causally relevant earlier B position.
            source = block_clean[int(layer)]
            def block_patch(_m, _i, out):
                is_tuple = isinstance(out, tuple)
                hidden = (out[0] if is_tuple else out).clone()
                hidden[batch_idx, successor_positions, :] = source[batch_idx, successor_positions, :].to(hidden.dtype)
                return (hidden, *out[1:]) if is_tuple else hidden
            h = item.block.register_forward_hook(block_patch)
            try:
                logits = adapter.forward(corrupt_ids).logits
            finally:
                h.remove()
            record(layer, "block_successor_position", "clean_patch", "block", None,
                   successor_positions.tolist(), logits)

            # Whole-block random-position patch is a location control.
            def random_block_patch(_m, _i, out):
                is_tuple = isinstance(out, tuple)
                hidden = (out[0] if is_tuple else out).clone()
                hidden[batch_idx, random_positions_t, :] = source[batch_idx, random_positions_t, :].to(hidden.dtype)
                return (hidden, *out[1:]) if is_tuple else hidden
            h = item.block.register_forward_hook(random_block_patch)
            try:
                logits = adapter.forward(corrupt_ids).logits
            finally:
                h.remove()
            record(layer, "block_random_position", "clean_patch", "block", None,
                   random_positions, logits)

            # Patch selected and activity-matched/random control attention head contexts.
            head_rows = candidates[(candidates.layer == int(layer)) & (candidates.kind == "head")]
            for _, candidate in head_rows.iterrows():
                head = int(candidate.component)
                start, stop = head * adapter.head_dim, (head + 1) * adapter.head_dim
                source_head = head_clean[int(layer)]
                def head_patch(_m, inputs, s=start, e=stop, src=source_head):
                    x = inputs[0].clone()
                    x[batch_idx, query_positions, s:e] = src[batch_idx, query_positions, s:e].to(x.dtype)
                    return (x, *inputs[1:])
                h = item.attention_output.register_forward_pre_hook(head_patch)
                try:
                    logits = adapter.forward(corrupt_ids).logits
                finally:
                    h.remove()
                record(layer, "head_context_final_query", str(candidate.selection), "head", head,
                       query_positions.tolist(), logits)

            # Patch selected and activity-matched control MLP neuron contributions.
            neuron_rows = candidates[(candidates.layer == int(layer)) & (candidates.kind == "neuron")]
            for _, candidate in neuron_rows.iterrows():
                neuron = int(candidate.component)
                source_mlp = mlp_clean[int(layer)]
                def neuron_patch(_m, inputs, j=neuron, src=source_mlp):
                    x = inputs[0].clone()
                    x[batch_idx, query_positions, j] = src[batch_idx, query_positions, j].to(x.dtype)
                    return (x, *inputs[1:])
                h = item.mlp_output.register_forward_pre_hook(neuron_patch)
                try:
                    logits = adapter.forward(corrupt_ids).logits
                finally:
                    h.remove()
                record(layer, "mlp_neuron_final_query", str(candidate.selection), "neuron", neuron,
                       query_positions.tolist(), logits)
    return pd.DataFrame(rows)


def summarize_activation_patching_v2(patching: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rows = []
    group_cols = ["layer", "patch_location", "kind", "selection", "component"]
    for key, group in patching.groupby(group_cols, dropna=False, sort=True):
        layer, location, kind, selection, component = key
        margin = _bootstrap_ci(group.margin_recovery.tolist(), seed + int(layer) * 31)
        logit = _bootstrap_ci(group.target_logit_recovery.tolist(), seed + int(layer) * 47 + 1)
        rows.append({
            "layer": int(layer), "patch_location": location, "kind": kind,
            "selection": selection, "component": component,
            "n": int(margin["n"]), "mean_margin_recovery": margin["mean"],
            "margin_recovery_ci95_low": margin["ci95_low"],
            "margin_recovery_ci95_high": margin["ci95_high"],
            "mean_target_logit_recovery": logit["mean"],
            "target_logit_recovery_ci95_low": logit["ci95_low"],
            "target_logit_recovery_ci95_high": logit["ci95_high"],
        })
    return pd.DataFrame(rows)


def summarize_copy_specificity(copy_details: pd.DataFrame, control_details: pd.DataFrame,
                                seed: int = 7, bootstrap_samples: int = 5000,
                                permutation_samples: int = 10000) -> pd.DataFrame:
    """Compare selected-vs-control ablation effects on copy vs random-transition tasks.

    Independent prompt sets are used, so the bootstrap and permutation are unpaired.
    """
    rng = np.random.default_rng(seed)
    output = []
    keys = ["kind", "layer", "pair_id"]
    for key, cg in copy_details.groupby(keys, sort=True):
        tg = control_details[(control_details.kind == key[0]) &
                             (control_details.layer == key[1]) &
                             (control_details.pair_id == key[2])]
        if tg.empty:
            continue
        cp = cg.pivot(index="prompt_id", columns="selection", values="copy_margin_drop")
        tp = tg.pivot(index="prompt_id", columns="selection", values="copy_margin_drop")
        if not {"discovery_top", "random_control"}.issubset(cp.columns):
            continue
        if not {"discovery_top", "random_control"}.issubset(tp.columns):
            continue
        c = (cp.discovery_top - cp.random_control).dropna().to_numpy(dtype=float)
        t = (tp.discovery_top - tp.random_control).dropna().to_numpy(dtype=float)
        if not len(c) or not len(t):
            continue
        observed = float(c.mean() - t.mean())
        cb = rng.choice(c, size=(bootstrap_samples, len(c)), replace=True).mean(axis=1)
        tb = rng.choice(t, size=(bootstrap_samples, len(t)), replace=True).mean(axis=1)
        diffs = cb - tb
        pooled = np.concatenate([c, t])
        n_c = len(c)
        null = np.empty(permutation_samples, dtype=float)
        for i in range(permutation_samples):
            perm = rng.permutation(pooled)
            null[i] = perm[:n_c].mean() - perm[n_c:].mean()
        p = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (permutation_samples + 1)
        output.append({
            "kind": key[0], "layer": int(key[1]), "pair_id": key[2],
            "n_copy_prompts": int(len(c)), "n_control_prompts": int(len(t)),
            "copy_minus_control_specificity": observed,
            "ci95_low": float(np.quantile(diffs, .025)),
            "ci95_high": float(np.quantile(diffs, .975)),
            "permutation_p": float(p),
            "copy_selected_vs_random_mean": float(c.mean()),
            "control_selected_vs_random_mean": float(t.mean()),
        })
    frame = pd.DataFrame(output)
    if not frame.empty:
        frame["q_bh"] = _benjamini_hochberg(frame.permutation_p.tolist())
    return frame
