import torch
from interpretability_lab.core import GPTConfig,TinyGPT,ActivationRecorder

def test_forward_hook_capture():
    cfg=GPTConfig(12,block_size=8,n_layer=1,n_head=2,n_embd=8)
    model=TinyGPT(cfg)
    recorder=ActivationRecorder(model).attach(["blocks.0.attn"])
    model(torch.randint(0,12,(1,5)))
    assert "blocks.0.attn" in recorder.activations
    assert recorder.activations["blocks.0.attn"].shape[:2]==(1,5)
    recorder.remove()
    assert len(recorder.handles)==0
