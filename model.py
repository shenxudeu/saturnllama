from typing import Optional, Tuple
from dataclasses import dataclass
from torch import nn
import torch
import math
from torch.nn import functional as F
import inspect


@dataclass
class ModelArgs:
    # default parameters as llama 7B.
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None 
    multiple_of: int = 256  # MLP hidden layer will be multiple of this number
    dropout: float = 0.0
    norm_eps: float = 1e-5
    max_seq_len: int = 2048

    flash = False  # If True, use Pytorch Attention instead of the manual implementation.

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Hyper Parameters
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads  ## TODO: what is n_kv_heads?
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1  # Across Only 1 GPU.
        self.n_local_heads = args.n_heads // model_parallel_size  # n_heads per GPU.
        self.n_local_kv_heads = args.n_kv_heads // model_parallel_size  # n_kv_heads per GPU.
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads  # Designed for n_kv_heads != n_heads
        
        # Weight Matrices
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # Query Matrix
        # self.wq = nn.Linear(args.dim, args.dim, bias=False)  # The same as above Matrix
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # Output Matrix

        # Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.res_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout  # For saving and loading.

        self.flash = args.flash

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)  # Upper triangular matrix

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.size()

        # Query, Key, Value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # (bsz, seqlen, dim) * 3
        # Split heads
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)  # (bsz, seqlen, n_local_heads, head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # (bsz, seqlen, n_local_kv_heads, head_dim)        
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # (bsz, seqlen, n_local_kv_heads, head_dim)

        # RoPE Embedding
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # group multiquery attention.
        xk = repeat_kv(xk, self.n_rep)  # (bsz, seqlen, n_local_kv_heads, head_dim) -> (bsz, seqlen, n_local_kv_heads * n_rep, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bsz, seqlen, n_local_kv_heads, head_dim) -> (bsz, seqlen, n_local_kv_heads * n_rep, head_dim)

        # make heads into a batch to apply attention.
        xq = xq.transpose(1, 2)  # (bsz, seqlen, n_local_heads, head_dim) -> (bsz, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Calculate Attention
        if self.flash:  # Use PyTorch Implemention.
            output = torch.nn.functional.sacale_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_casual=True)
        else:  # Use Manual Implemention.
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bsz, n_local_heads, seqlen, head_dim)

        # restore original shape and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # (bsz, seqlen, n_local_heads, head_dim) -> (bsz, seqlen, dim)
        
        # Output Matrix
        output = self.wo(output)  # (bsz, seqlen, dim)
        output = self.res_dropout(output)
        return output
    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        # Attention
        h = x + self.attention_norm(self.attention(x, freqs_cos, freqs_sin))

        # Feed Forward
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        out = h + self.ffn_norm(self.feed_forward(x))
        return out

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_emb = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_emb.weight = self.output.weight

        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # init weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        self.last_loss = None

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.size()
        h = self.tok_emb(tokens)  # (bsz, seqlen, dim)
        h = self.dropout(h)
        freqs_cos, freqs_sin = self.freqs_cos[:seqlen], self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean")
        else:
            # only forward the output on the very last position.
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no weight decay.
        # i.e. all weight tensors in matmuls + embeddings decay, all bias and layernorms do not decay.
        decay_params = [p for n, p in param_dict.items() if len(p.shape) >= 2]
        nondecay_params = [p for n, p in param_dict.items() if len(p.shape) < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        print(f"number of params: {num_decay_params + num_nondecay_params}")
        print(f"number of decay params: {num_decay_params}")
        print(f"number of non-decay params: {num_nondecay_params}")

        # create optimizer
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused adam: {use_fused}")

        return optimizer