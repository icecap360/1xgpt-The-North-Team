from torch import nn, Tensor
from einops import rearrange

from genie.attention import SelfAttention
from genie.st_moe_pytorch import MoE, Mlp



class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # sequence dim is over each frame's 16x16 patch tokens
        self.spatial_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )

        # sequence dim is over time sequence (16)
        self.temporal_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )
        
        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.mlp = MoE(
            d_model, 
            num_experts=4,
            expert_hidden_mult=1.0,
            gating_top_n=2
            )
        
    # def forward(self, x_TSC: Tensor) -> Tensor:
    #     # Process attention spatially
    #     T, S = x_TSC.size(1), x_TSC.size(2)
    #     x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
    #     x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

    #     # Process attention temporally
    #     x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
    #     x_TC = x_TC + self.temporal_attn(x_TC, causal=True)

    #     # Apply the MLP
    #     x_TC = x_TC + self.mlp(self.norm2(x_TC))
    #     x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
    #     return x_TSC 
    def forward(self, x_TSC: Tensor) -> Tensor:
        # Process attention spatially
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        # Process attention temporally
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)

        # Apply the MLP
        moe_outputs = self.mlp(self.norm2(x_TC))
        x_TC = x_TC + moe_outputs.outputs
        x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        return x_TSC, moe_outputs.total_aux_loss

class STTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([STBlock(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
            mlp_ratio=mlp_ratio,
            mlp_bias=mlp_bias,
            mlp_drop=mlp_drop,
        ) for _ in range(num_layers)])

    # def forward(self, tgt):
    #     x = tgt
    #     for layer in self.layers:
    #         x = layer(x)
    #     return x
    def forward(self, tgt):
        x = tgt
        moe_loss = None
        for layer in self.layers:
            x, aux_loss = layer(x)
            if moe_loss == None:
                moe_loss = aux_loss
            else:
                moe_loss += aux_loss
        return x, moe_loss