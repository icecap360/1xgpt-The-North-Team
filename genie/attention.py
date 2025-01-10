import torch
from torch import nn
from xformers.ops import LowerTriangularMask, memory_efficient_attention, unbind
import os


XFORMERS_DISABLED = os.environ.get("XFORMERS_DISABLED", "false").lower() == "true"

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

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

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
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        self.q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N1, C = x1.shape
        _, N2, _ = x2.shape

        q = self.q(x1) #B, N, C
        k = self.k(x2) #B, S, C
        v = self.v(x2) #B, S, C

        q = q.reshape(B, N1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, N2, head_dim)
        v = v.reshape(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, N2, head_dim)

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

        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N1, C)
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

import genie.nGPT as nGPT
if XFORMERS_DISABLED:
    SelfAttention = BasicSelfAttention
    CrossAttention = BasicCrossAttention
else:
    SelfAttention = BasicSelfAttention
    CrossAttention = BasicCrossAttention
    # SelfAttention = MemoryEfficientAttention
    # CrossAttention = MemoryEfficientCrossAttention