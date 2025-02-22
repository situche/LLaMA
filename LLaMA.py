import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import 可选
import math

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[在t] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[在t] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_length: int = 2048
    device: str = None

# 旋转式位置编码  theta=θ
# 预计算频率
def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    assert head_dim % 2 == 0, '根据论文要求维度必须为偶数'
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).到(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

# 应用旋转嵌入
def apply_rotary_embeddings(x, freqs_complex, device):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).到(device)

# RMS归一化
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

# 自注意力机制
def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_q_heads = args.n_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads    

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.head_dim))

    def forward(self, x, start_pos, freqs_complex):
        batch_size, seq_len, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output)

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
            self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
            self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        x_ = self.w1(x)
        swish = F.silu(x_)
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

# 编码块
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
# LLaMA
class llamaModel(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size!= -1, "词汇表大小必须设置"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "一次只能处理一个token"

        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
