"""Reusable experiment utilities for reproducible mechanistic studies."""
from __future__ import annotations

import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_texts(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def environment_metadata(seed: int, model_name: str, prompts: Sequence[str]) -> dict[str, Any]:
    """Capture enough run provenance to reproduce an experiment."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "model": model_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "prompt_count": len(prompts),
        "prompt_sha256": sha256_texts(prompts),
    }


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def bootstrap_mean(values: Sequence[float], *, seed: int = 7, samples: int = 2000) -> dict[str, float | int]:
    """Percentile bootstrap CI for the mean across independent prompt-level observations."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot summarize an empty or non-finite sample")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(samples, x.size), replace=True).mean(axis=1)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }
