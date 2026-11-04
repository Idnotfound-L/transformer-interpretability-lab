from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

DEMO_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "transformers process sequences using attention. "
    "a model predicts the next token from previous context. "
    "mechanistic interpretability studies internal computation. "
) * 100

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 96
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")

class CharTokenizer:
    def __init__(self, text="", chars=None):
        """Character tokenizer; pass ``chars`` to restore a saved vocabulary."""
        self.chars = list(chars) if chars is not None else sorted(set(text))
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("Tokenizer vocabulary contains duplicate characters")
        if not self.chars:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for c,i in self.stoi.items()}
    @property
    def vocab_size(self):
        return len(self.chars)
    def encode(self, text):
        missing = [c for c in text if c not in self.stoi]
        if missing:
            raise ValueError(f"Characters absent from vocabulary: {missing[:5]}")
        return [self.stoi[c] for c in text]
    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size + 1:
        raise ValueError("Dataset must contain more than block_size + 1 tokens")
    ix = torch.randint(len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3*cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1,1,cfg.block_size,cfg.block_size), persistent=False)
        self.last_attention = None
        self.last_head_outputs = None
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(C, dim=-1)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.last_attention = attn.detach()
        head_out = self.attn_dropout(attn) @ v
        self.last_head_outputs = head_out.detach()
        y = head_out.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_dropout(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4*cfg.n_embd)
        self.proj = nn.Linear(4*cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0, 0.02)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        if T > self.config.block_size:
            raise ValueError("Input exceeds configured context length")
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None,:,:]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)), targets.reshape(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits,_ = self(idx[:,-self.config.block_size:])
            logits = logits[:,-1,:] / max(temperature,1e-5)
            if top_k:
                values,_ = torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits < values[:,-1,None]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits,dim=-1),1)],dim=1)
        return idx

class ActivationRecorder:
    def __init__(self, model):
        self.model, self.activations, self.handles = model, {}, []
    def attach(self, names=None):
        self.remove()
        modules = dict(self.model.named_modules())
        names = names or [n for n in modules if n.endswith(".attn") or n.endswith(".mlp")]
        for name in names:
            if name in modules:
                def hook(module, inputs, output, key=name):
                    value = output[0] if isinstance(output,tuple) else output
                    self.activations[key] = value.detach().cpu()
                self.handles.append(modules[name].register_forward_hook(hook))
        return self
    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []
    def __enter__(self): return self
    def __exit__(self, *args): self.remove()

@torch.no_grad()
def collect_attention(model, ids):
    model.eval(); model(ids)
    return [b.attn.last_attention.detach().cpu() for b in model.blocks]

@torch.no_grad()
def collect_residuals(model, ids):
    model.eval()
    x = model.token_embedding(ids) + model.position_embedding(torch.arange(ids.shape[1],device=ids.device))[None,:,:]
    values = [x.detach().cpu()]
    for block in model.blocks:
        x = x + block.attn(block.ln1(x))
        x = x + block.mlp(block.ln2(x))
        values.append(x.detach().cpu())
    return values

@torch.no_grad()
def collect_mlp_hidden(model, ids):
    model.eval()
    x = model.token_embedding(ids) + model.position_embedding(torch.arange(ids.shape[1],device=ids.device))[None,:,:]
    outputs = []
    for block in model.blocks:
        x = x + block.attn(block.ln1(x))
        h = F.gelu(block.mlp.fc(block.ln2(x)))
        outputs.append(h.detach().cpu())
        x = x + block.mlp.proj(h)
    return outputs

def pca_projection(reps):
    a = reps.detach().cpu().numpy()
    if a.ndim == 3: a = a[0]
    return PCA(n_components=min(2,a.shape[0],a.shape[1])).fit_transform(a)

def attention_entropy(attn):
    p = attn.clamp_min(1e-12)
    return -(p*p.log()).sum(-1).mean((0,2))

def mean_attention_distance(attn):
    _,_,q,k = attn.shape
    qi = torch.arange(q).view(1,1,q,1)
    ki = torch.arange(k).view(1,1,1,k)
    return (attn.cpu()*(qi-ki).abs()).sum(-1).mean((0,2))

@contextmanager
def ablate_head(model, layer, head):
    module = model.blocks[layer].attn
    original = module.forward
    def changed(x):
        B,T,C = x.shape
        q,k,v = module.qkv(x).split(C,dim=-1)
        q=q.view(B,T,module.n_head,module.head_dim).transpose(1,2)
        k=k.view(B,T,module.n_head,module.head_dim).transpose(1,2)
        v=v.view(B,T,module.n_head,module.head_dim).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/(module.head_dim**0.5)
        scores=scores.masked_fill(module.mask[:,:,:T,:T]==0,float("-inf"))
        attn=F.softmax(scores,dim=-1)
        module.last_attention=attn.detach()
        out=attn@v
        out[:,head]=0
        module.last_head_outputs=out.detach()
        y=out.transpose(1,2).contiguous().view(B,T,C)
        return module.resid_dropout(module.proj(y))
    module.forward=changed
    try: yield
    finally: module.forward=original

@contextmanager
def ablate_neuron(model, layer, neuron):
    mlp=model.blocks[layer].mlp
    original=mlp.forward
    def changed(x):
        h=F.gelu(mlp.fc(x)).clone()
        h[...,neuron]=0
        return mlp.dropout(mlp.proj(h))
    mlp.forward=changed
    try: yield
    finally: mlp.forward=original

@torch.no_grad()
def run_ablation(model, ids, kind, layer, component):
    model.eval()
    base,_=model(ids)
    base_probs=base[:,-1].softmax(-1)
    context=ablate_head(model,layer,component) if kind=="head" else ablate_neuron(model,layer,component)
    with context:
        altered,_=model(ids)
    p=base_probs.clamp_min(1e-12)
    q=altered[:,-1].softmax(-1).clamp_min(1e-12)
    return {
        "baseline_top_probability": float(base_probs.max()),
        "altered_top_probability": float(q.max()),
        "kl_baseline_to_altered": float((p*(p.log()-q.log())).sum()),
        "baseline_top_id": int(base_probs.argmax()),
        "altered_top_id": int(q.argmax()),
        "baseline_probs": base_probs.cpu(),
        "altered_probs": q.cpu()
    }
