import argparse
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prompt",default="The capital of France is")
    a=p.parse_args()
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        raise SystemExit("Install optional dependencies: pip install -r requirements-gpt2.txt")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=HookedTransformer.from_pretrained("gpt2-small",device=device)
    logits,cache=model.run_with_cache(a.prompt)
    probs=logits[0,-1].softmax(-1)
    values,ids=probs.topk(10)
    for token_id,prob in zip(ids.tolist(),values.tolist()):
        print(repr(model.to_string(token_id)),f"{prob:.4f}")
    print("\\nCached activation names:")
    for name in list(cache.keys())[:30]: print(name)

if __name__=="__main__": main()
