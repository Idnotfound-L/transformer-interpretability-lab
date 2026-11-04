"""Optional GPT-2 Small induction-pattern probe using TransformerLens.

Install optional dependencies with: pip install -r requirements-gpt2.txt
This script reports descriptive attention scores only; it does not claim to
identify a functional induction head without causal validation.
"""
import argparse
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise SystemExit("Install optional requirements: pip install -r requirements-gpt2.txt") from exc

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    prompts = [
        "A B C D A",
        "red blue green yellow red",
        "cat dog bird fish cat",
        "one two three four one",
    ]
    rows = []
    for text in prompts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        # Locate the first and final occurrence of the first token.
        token_ids = tokens[0].tolist()
        first = token_ids[0]
        positions = [i for i, token in enumerate(token_ids) if token == first]
        if len(positions) < 2:
            continue
        q, k = positions[-1], positions[0]
        pattern = cache["pattern"]  # [layer, batch, head, query, key]
        scores = pattern[ :, 0, :, q, k].detach().cpu()
        for layer in range(scores.shape[0]):
            for head in range(scores.shape[1]):
                rows.append({"prompt": text, "layer": layer, "head": head,
                             "first_to_last_repeat_attention": float(scores[layer, head])})
    if not rows:
        raise SystemExit("No repeated-token prompt was successfully tokenized; adjust prompts.")
    df = pd.DataFrame(rows)
    summary = df.groupby(["layer", "head"], as_index=False)["first_to_last_repeat_attention"].mean()
    summary.to_csv("gpt2_induction_probe.csv", index=False)
    print(summary.sort_values("first_to_last_repeat_attention", ascending=False).head(15).to_string(index=False))
    print("Saved gpt2_induction_probe.csv. High attention is a candidate pattern, not causal proof.")

if __name__ == "__main__":
    main()
