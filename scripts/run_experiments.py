import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretability_lab.core import GPTConfig, CharTokenizer, TinyGPT, DEMO_TEXT, collect_attention, attention_entropy, mean_attention_distance, run_ablation

def main():
    ck=torch.load(ROOT/"artifacts"/"tiny_gpt.pt",map_location="cpu",weights_only=False)
    tok=CharTokenizer(chars=ck.get("chars", sorted(set(DEMO_TEXT))))
    cfg=GPTConfig(**ck["config"])
    model=TinyGPT(cfg); model.load_state_dict(ck["model_state"])
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device).eval()
    prompt="the quick"
    ids=torch.tensor([tok.encode(prompt)],device=device)
    attentions=collect_attention(model,ids)
    summary=[]
    for layer,a in enumerate(attentions):
        summary.append({"layer":layer,"entropy":attention_entropy(a).tolist(),"mean_distance":mean_attention_distance(a).tolist()})
    ab=run_ablation(model,ids,"head",0,0)
    out={"prompt":prompt,"attention_summary":summary,
         "head_ablation_layer0_head0":{k:v for k,v in ab.items() if not k.endswith("_probs")},
         "caution":"Single-prompt demonstration; do not generalise from this result."}
    (ROOT/"artifacts").mkdir(exist_ok=True)
    (ROOT/"artifacts"/"experiment_summary.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
