import torch
from torch import nn
from xformers.ops import LowerTriangularMask, memory_efficient_attention, unbind
import os
from functools import partial


XFORMERS_DISABLED = os.environ.get("XFORMERS_DISABLED", "false").lower() == "true"

def init_t_xy(seq_len: int):
    end_x = end_y = int(seq_len**0.5)
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    
    return t_x, t_y

def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x ** 2)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def compute_cis_1d(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # type: ignore
    freqs = torch.outer(t, freqs)  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast_2d(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    # print (freqs_cis.shape, x.shape)
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def reshape_for_broadcast_1d(freqs_cis: torch.Tensor):
    # broadcast over all batches and heads
    # x has shape [B, heads, T, head_dim / 2]
    # freq_cis has shape [T, head_dim/2] 
    return freqs_cis.unsqueeze(0).unsqueeze(0)

def apply_rotary_emb_2d(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    
    # print (xq.float().reshape(*xq.shape[:-1], -1, 2).shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous())
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous())
    freqs_cis = reshape_for_broadcast_2d(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def apply_rotary_emb_1d(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    if not torch.cuda.is_available():
        xq = xq.to('cpu')
        xk = xk.to('cpu')
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous())
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous())
    freqs_cis = reshape_for_broadcast_1d(freqs_cis)
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(xq.shape)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(xk.shape)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class BasicSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        rope=True,
        is_2d = True,
        rope_mixed=False,
        seq_len=16
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)
        
        self.is_2d = is_2d
        self.rope_mixed = rope_mixed
        self.rope = rope

        if rope:
            if self.is_2d:
                if self.rope_mixed:
                    self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
                    
                    freqs = init_2d_freqs(
                        dim=d_model // self.num_heads, num_heads=self.num_heads, theta=10.0, 
                        rotate=True
                    ).view(2, -1)
                    self.freqs = nn.Parameter(freqs, requires_grad=True)
                    
                    t_x, t_y = init_t_xy(256)
                    self.register_buffer('rope_t_x', t_x)
                    self.register_buffer('rope_t_y', t_y)

                else:
                    self.compute_cis = partial(compute_axial_cis, dim=d_model // self.num_heads, theta=10.0)
                    freqs_cis = self.compute_cis(end_x=16, end_y=16)
                    self.rope_freqs_cis = freqs_cis

            else:
                self.rope_theta = 1000
                self.rope_freqs_cis = compute_cis_1d(
                    self.head_dim,
                    seq_len,
                    self.rope_theta,
                )

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            if self.is_2d:
                # # print (N, t_x.shape, t_y.shape)

                if self.rope_mixed:
                    t_x, t_y = self.rope_t_x, self.rope_t_y
                    freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
                else:
                    freqs_cis = self.rope_freqs_cis.to(x.device)
                
                if q.shape[1] == 256:
                    q, k = apply_rotary_emb_2d(q, k, freqs_cis)
                else:
                    # only apply rotary emb on the video tokens
                    q[:, :256, :], k[:, :256, :] = apply_rotary_emb_2d(q[:, :256, :], k[:, :256, :], freqs_cis)

                assert not (torch.isnan(q).any() or torch.isnan(k).any()), "NaN detected in RoPE output"
            else:
                q, k = apply_rotary_emb_1d(q, k, freqs_cis=self.rope_freqs_cis.to(x.device))

        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MemoryEfficientAttention(BasicSelfAttention):
    # NOTE: Mem-eff attention from xformers is actually Flash Attention 2
        
    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = unbind(qkv, 2)
        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)    
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        attn_bias = LowerTriangularMask() if causal else None
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        return x

class BasicCrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        d_kv=None
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        self.q = nn.Linear(d_model, d_model, bias=qkv_bias)
        if not d_kv:
            d_kv = d_model
        self.k = nn.Linear(d_kv, d_model, bias=qkv_bias)
        self.v = nn.Linear(d_kv, d_model, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x1.shape

        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)

        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        q *= self.scale
        attn = q @ k.transpose(-2, -1)
        

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MemoryEfficientCrossAttention(BasicSelfAttention):
    # NOTE: Mem-eff attention from xformers is actually Flash Attention 2
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)    
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        attn_bias = LowerTriangularMask() if causal else None
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        return x

# import genie.nGPT as nGPT
if XFORMERS_DISABLED:
    SelfAttention = BasicSelfAttention
    CrossAttention = BasicCrossAttention
else:
    SelfAttention = BasicSelfAttention
    CrossAttention = BasicCrossAttention
    # SelfAttention = MemoryEfficientAttention
    # CrossAttention = MemoryEfficientCrossAttention