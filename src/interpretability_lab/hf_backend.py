"""Architecture-aware instrumentation for Hugging Face decoder-only Transformers.

The adapter deliberately intervenes at projection inputs. For Llama-like/GQA models,
``o_proj`` receives the concatenated *query-head* context vectors, so slicing this
input by ``hidden_size // num_attention_heads`` removes one query head even when
key/value heads are shared. ``down_proj`` receives the gated MLP intermediate vector,
which makes its input a natural point for neuron-level interventions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LayerModules:
    index: int
    block: nn.Module
    attention: nn.Module
    attention_output: nn.Module
    mlp: nn.Module
    mlp_output: nn.Module
    block_path: str
    attention_output_path: str
    mlp_output_path: str


class HuggingFaceTransformerAdapter:
    """Inspect and intervene on common Hugging Face causal-LM architectures."""

    def __init__(self, model: nn.Module, tokenizer: Any, model_id: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = next(model.parameters()).device
        self.config = model.config
        self.layers = self._resolve_layers()
        self.n_layers = len(self.layers)
        self.n_heads = int(
            getattr(self.config, "num_attention_heads", None)
            or getattr(self.config, "n_head", 0)
        )
        hidden = int(
            getattr(self.config, "hidden_size", None)
            or getattr(self.config, "n_embd", 0)
        )
        if self.n_heads <= 0 or hidden <= 0:
            raise ValueError("Could not infer num_attention_heads and hidden size from model.config")
        self.hidden_size = hidden
        self.head_dim = int(getattr(self.config, "head_dim", hidden // self.n_heads))
        self.intermediate_size = self._infer_intermediate_size()
        self.n_kv_heads = int(
            getattr(self.config, "num_key_value_heads", None) or self.n_heads
        )
        self.architecture = type(model).__name__

    @staticmethod
    def _find_attr(module: nn.Module, names: Iterable[str]) -> tuple[str, nn.Module]:
        for name in names:
            child = getattr(module, name, None)
            if isinstance(child, nn.Module):
                return name, child
        raise ValueError(
            f"Could not find any of {tuple(names)} on module {type(module).__name__}"
        )

    def _resolve_layers(self) -> list[LayerModules]:
        candidates: list[tuple[str, nn.ModuleList]] = []
        for path, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module):
                if any(hasattr(module[0], n) for n in ("self_attn", "attn", "attention")):
                    candidates.append((path, module))
        if not candidates:
            raise ValueError(
                "Unsupported HF architecture: no decoder ModuleList with self_attn/attn found. "
                "Expected a Llama-like `model.layers` or GPT-2-like `transformer.h` stack."
            )
        # Prefer conventional decoder stacks over nested auxiliary module lists.
        candidates.sort(key=lambda item: (not item[0].endswith((".layers", ".h")), len(item[0])))
        stack_path, stack = candidates[0]
        resolved: list[LayerModules] = []
        for index, block in enumerate(stack):
            attn_attr, attention = self._find_attr(block, ("self_attn", "attn", "attention"))
            mlp_attr, mlp = self._find_attr(block, ("mlp", "feed_forward", "ffn"))
            attn_out_attr, attn_out = self._find_attr(
                attention, ("o_proj", "out_proj", "c_proj", "dense")
            )
            mlp_out_attr, mlp_out = self._find_attr(
                mlp, ("down_proj", "c_proj", "dense_4h_to_h", "fc2")
            )
            block_path = f"{stack_path}.{index}"
            resolved.append(
                LayerModules(
                    index=index,
                    block=block,
                    attention=attention,
                    attention_output=attn_out,
                    mlp=mlp,
                    mlp_output=mlp_out,
                    block_path=block_path,
                    attention_output_path=f"{block_path}.{attn_attr}.{attn_out_attr}",
                    mlp_output_path=f"{block_path}.{mlp_attr}.{mlp_out_attr}",
                )
            )
        return resolved

    def _infer_intermediate_size(self) -> int:
        value = getattr(self.config, "intermediate_size", None)
        if value:
            return int(value)
        value = getattr(self.config, "n_inner", None)
        if value:
            return int(value)
        value = getattr(self.config, "n_embd", None)
        if value:
            return int(value) * 4
        # Infer from a conventional gated MLP projection if model config omits it.
        first = self.layers[0].mlp
        for name in ("up_proj", "gate_proj", "fc1"):
            layer = getattr(first, name, None)
            if isinstance(layer, nn.Linear):
                return int(layer.out_features)
        raise ValueError("Could not infer the MLP intermediate width")

    def encode(self, text: str, *, add_special_tokens: bool = True) -> torch.Tensor:
        if not text or not text.strip():
            raise ValueError("Prompt cannot be empty")
        encoded = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        )
        return encoded["input_ids"].to(self.device)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, *, output_attentions: bool = False):
        self.model.eval()
        return self.model(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=output_attentions,
            return_dict=True,
        )

    def _check_layer(self, layer: int) -> LayerModules:
        if not 0 <= int(layer) < self.n_layers:
            raise IndexError(f"Layer must be in [0, {self.n_layers - 1}]")
        return self.layers[int(layer)]

    @contextmanager
    def ablate_head(self, layer: int, head: int, position: int | None = None):
        """Zero one query-head context vector before the output projection.

        If ``position`` is supplied, intervention is restricted to that sequence
        position (negative indices are supported). This is useful for causal
        next-token studies because only the final query position directly feeds
        the prediction being measured.
        """
        item = self._check_layer(layer)
        if not 0 <= int(head) < self.n_heads:
            raise IndexError(f"Head must be in [0, {self.n_heads - 1}]")
        start, stop = int(head) * self.head_dim, (int(head) + 1) * self.head_dim

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError("Attention output projection did not receive a tensor input")
            x = inputs[0].clone()
            if x.shape[-1] != self.hidden_size:
                raise RuntimeError(
                    f"Expected attention projection input width {self.hidden_size}, got {x.shape[-1]}"
                )
            if position is None:
                x[..., start:stop] = 0
            else:
                pos = int(position)
                if pos < 0:
                    pos += x.shape[-2]
                if not 0 <= pos < x.shape[-2]:
                    raise IndexError(f"Position {position} is outside sequence length {x.shape[-2]}")
                x[:, pos, start:stop] = 0
            return (x, *inputs[1:])

        handle = item.attention_output.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def ablate_neuron(self, layer: int, neuron: int, position: int | None = None):
        """Zero one post-gating MLP dimension before down projection.

        If ``position`` is supplied, intervention is restricted to that sequence
        position; otherwise the coordinate is zeroed at every position.
        """
        item = self._check_layer(layer)
        if not 0 <= int(neuron) < self.intermediate_size:
            raise IndexError(f"Neuron must be in [0, {self.intermediate_size - 1}]")

        def hook(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError("MLP down projection did not receive a tensor input")
            x = inputs[0].clone()
            if x.shape[-1] != self.intermediate_size:
                raise RuntimeError(
                    "MLP down-projection input width did not match inferred intermediate size: "
                    f"expected {self.intermediate_size}, got {x.shape[-1]}"
                )
            if position is None:
                x[..., int(neuron)] = 0
            else:
                pos = int(position)
                if pos < 0:
                    pos += x.shape[-2]
                if not 0 <= pos < x.shape[-2]:
                    raise IndexError(f"Position {position} is outside sequence length {x.shape[-2]}")
                x[:, pos, int(neuron)] = 0
            return (x, *inputs[1:])

        handle = item.mlp_output.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def ablation_metrics(self, input_ids: torch.Tensor, kind: str, layer: int, component: int) -> dict:
        baseline = self.forward(input_ids).logits[:, -1, :].float()
        intervention = self.ablate_head(layer, component) if kind == "head" else self.ablate_neuron(layer, component)
        with intervention:
            altered = self.forward(input_ids).logits[:, -1, :].float()
        p = baseline.softmax(-1).clamp_min(1e-12)
        q = altered.softmax(-1).clamp_min(1e-12)
        baseline_top = int(baseline.argmax(-1).item())
        return {
            "model": self.model_id,
            "kind": kind,
            "layer": int(layer),
            "component": int(component),
            "baseline_top_token_id": baseline_top,
            "altered_top_token_id": int(altered.argmax(-1).item()),
            "baseline_top_probability": float(p.max().item()),
            "altered_probability_of_baseline_top": float(q[0, baseline_top].item()),
            "kl_baseline_to_intervened": float((p * (p.log() - q.log())).sum(-1).mean().item()),
            "baseline_top_logit_change": float((altered[0, baseline_top] - baseline[0, baseline_top]).item()),
            "top_token_changed": bool(baseline_top != int(altered.argmax(-1).item())),
        }

    @torch.inference_mode()
    def capture_mlp_intermediates(self, input_ids: torch.Tensor, layer_indices: Iterable[int]) -> dict[int, torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layer_indices:
            item = self._check_layer(int(layer))

            def hook(_module, inputs, key=int(layer)):
                if inputs and torch.is_tensor(inputs[0]):
                    captured[key] = inputs[0].detach().float().cpu()

            handles.append(item.mlp_output.register_forward_pre_hook(hook))
        try:
            self.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    @torch.inference_mode()
    def activation_patch_sweep(
        self, clean_ids: torch.Tensor, corrupted_ids: torch.Tensor, layer_count: int | None = None
    ) -> list[dict]:
        """Patch clean block output at the final token into a corrupted run, one layer at a time."""
        if clean_ids.shape != corrupted_ids.shape:
            raise ValueError(
                "Clean and corrupted prompts must tokenize to identical shapes for aligned patching; "
                f"got {tuple(clean_ids.shape)} and {tuple(corrupted_ids.shape)}"
            )
        n_layers = min(layer_count or self.n_layers, self.n_layers)
        clean_out = self.forward(clean_ids)
        corrupt_out = self.forward(corrupted_ids)
        clean_logits = clean_out.logits[0, -1].float()
        corrupt_logits = corrupt_out.logits[0, -1].float()
        target_id = int(clean_logits.argmax().item())
        clean_target = float(clean_logits[target_id].item())
        corrupt_target = float(corrupt_logits[target_id].item())
        denominator = clean_target - corrupt_target
        rows = []

        for layer in range(n_layers):
            item = self.layers[layer]
            clean_activation: dict[str, torch.Tensor] = {}

            def save_clean(_module, _inputs, output):
                first = output[0] if isinstance(output, tuple) else output
                clean_activation["value"] = first.detach()

            save_handle = item.block.register_forward_hook(save_clean)
            try:
                self.forward(clean_ids)
            finally:
                save_handle.remove()
            source = clean_activation["value"]

            def patch(_module, _inputs, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                hidden = hidden.clone()
                hidden[:, -1, :] = source[:, -1, :]
                return (hidden, *output[1:]) if is_tuple else hidden

            patch_handle = item.block.register_forward_hook(patch)
            try:
                patched_logits = self.forward(corrupted_ids).logits[0, -1].float()
            finally:
                patch_handle.remove()

            # Capture and reinsert the corrupted activation as a self-patch control.
            corrupt_activation: dict[str, torch.Tensor] = {}

            def save_corrupt(_module, _inputs, output):
                first = output[0] if isinstance(output, tuple) else output
                corrupt_activation["value"] = first.detach()

            corrupt_save_handle = item.block.register_forward_hook(save_corrupt)
            try:
                self.forward(corrupted_ids)
            finally:
                corrupt_save_handle.remove()
            corrupt_source = corrupt_activation["value"]

            def control(_module, _inputs, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                hidden = hidden.clone()
                hidden[:, -1, :] = corrupt_source[:, -1, :]
                return (hidden, *output[1:]) if is_tuple else hidden

            control_handle = item.block.register_forward_hook(control)
            try:
                control_logits = self.forward(corrupted_ids).logits[0, -1].float()
            finally:
                control_handle.remove()

            patched_target = float(patched_logits[target_id].item())
            p = clean_logits.softmax(-1).clamp_min(1e-12)
            q = patched_logits.softmax(-1).clamp_min(1e-12)
            rows.append({
                "layer": layer,
                "target_token_id": target_id,
                "clean_target_logit": clean_target,
                "corrupted_target_logit": corrupt_target,
                "corrupted_self_patch_control_logit": float(control_logits[target_id].item()),
                "clean_activation_patched_logit": patched_target,
                "logit_recovery_fraction": (patched_target - corrupt_target) / denominator if abs(denominator) > 1e-12 else float("nan"),
                "kl_clean_to_patched": float((p * (p.log() - q.log())).sum().item()),
            })
        return rows


def load_huggingface_model(
    model_id: str = "HuggingFaceTB/SmolLM3-3B-Base",
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    revision: str | None = None,
) -> HuggingFaceTransformerAdapter:
    """Load a pretrained causal LM. Transformers is an optional dependency."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face support is optional. Install it with `pip install -r requirements-huggingface.txt`."
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    if dtype == "auto":
        if device == "cuda":
            dtype_obj = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype_obj = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose auto, float32, float16 or bfloat16")
        dtype_obj = dtype_map[dtype]

    load_kwargs = {"local_files_only": local_files_only}
    if revision is not None and revision.strip():
        load_kwargs["revision"] = revision.strip()
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    # Eager attention is requested so `output_attentions=True` can return per-head patterns.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_obj,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    model.to(device)
    model.eval()
    return HuggingFaceTransformerAdapter(model, tokenizer, model_id=model_id)
