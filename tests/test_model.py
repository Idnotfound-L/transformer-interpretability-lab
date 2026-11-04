import torch
from interpretability_lab.core import GPTConfig,TinyGPT

def test_shapes_and_loss():
    cfg=GPTConfig(20,block_size=12,n_layer=2,n_head=2,n_embd=16)
    model=TinyGPT(cfg)
    x=torch.randint(0,20,(2,8)); y=torch.randint(0,20,(2,8))
    logits,loss=model(x,y)
    assert logits.shape==(2,8,20)
    assert loss.ndim==0

def test_causal_mask():
    cfg=GPTConfig(20,block_size=12,n_layer=1,n_head=2,n_embd=16,dropout=0)
    model=TinyGPT(cfg).eval()
    model(torch.randint(0,20,(1,8)))
    attn=model.blocks[0].attn.last_attention
    future=torch.triu(torch.ones(8,8),diagonal=1).bool()
    assert torch.all(attn[0,:,future]==0)


def test_ablation_context_restores_original_forward():
    from interpretability_lab.core import ablate_head
    cfg=GPTConfig(20, block_size=8, n_layer=1, n_head=2, n_embd=8, dropout=0.0)
    model=TinyGPT(cfg).eval()
    x=torch.randint(0,20,(1,5))
    original=model.blocks[0].attn.forward
    with ablate_head(model, 0, 0):
        assert model.blocks[0].attn.forward is not original
    assert model.blocks[0].attn.forward.__func__ is original.__func__
