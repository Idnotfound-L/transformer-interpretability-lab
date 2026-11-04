"""Train TinyGPT for pipeline validation or on a user-supplied text corpus."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interpretability_lab.core import GPTConfig, CharTokenizer, TinyGPT, DEMO_TEXT, get_batch
from interpretability_lab.experiment import seed_everything


def estimate_loss(model, train, val, block_size, batch_size, device, batches=20):
    model.eval()
    result = {}
    with torch.no_grad():
        for name, split in (("train", train), ("validation", val)):
            losses = []
            for _ in range(batches):
                x, y = get_batch(split, batch_size, block_size, device)
                _, loss = model(x, y)
                losses.append(float(loss))
            result[name] = sum(losses) / len(losses)
    model.train()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--corpus", type=Path, default=None, help="UTF-8 text file; default is a tiny smoke-test corpus")
    p.add_argument("--output", type=Path, default=Path("artifacts/tiny_gpt.pt"))
    a = p.parse_args()
    if a.steps < 1 or a.batch_size < 1 or a.block_size < 4:
        p.error("steps, batch-size must be positive and block-size must be >= 4")

    seed_everything(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = a.corpus.read_text(encoding="utf-8") if a.corpus else DEMO_TEXT
    if len(text) < 100:
        p.error("Corpus is too small; provide at least 100 characters")
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(encoded))
    train, val = encoded[:split], encoded[split:]
    if min(len(train), len(val)) <= a.block_size + 1:
        p.error("Corpus is too short for the selected block-size; use a longer corpus or smaller block-size")

    cfg = GPTConfig(tokenizer.vocab_size, block_size=a.block_size)
    model = TinyGPT(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=a.learning_rate)
    history = []
    for step in trange(1, a.steps + 1, desc="Training TinyGPT"):
        model.train()
        x, y = get_batch(train, a.batch_size, a.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % a.eval_interval == 0 or step == a.steps:
            metrics = estimate_loss(model, train, val, a.block_size, a.batch_size, device, a.eval_batches)
            metrics["step"] = step
            history.append(metrics)
            print(f"step={step} train_loss={metrics['train']:.4f} validation_loss={metrics['validation']:.4f}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "config": cfg.__dict__,
        "chars": tokenizer.chars,
        "seed": a.seed,
        "training": {
            "steps": a.steps, "batch_size": a.batch_size, "learning_rate": a.learning_rate,
            "block_size": a.block_size, "corpus": str(a.corpus) if a.corpus else "built-in smoke-test corpus",
            "corpus_characters": len(text), "split": "first 90% train / final 10% validation",
            "history": history, "python": platform.python_version(), "torch": torch.__version__,
        },
    }
    torch.save(checkpoint, a.output)
    metrics_path = a.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(checkpoint["training"], indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {a.output}")
    print(f"Saved training metrics: {metrics_path}")


if __name__ == "__main__":
    main()
