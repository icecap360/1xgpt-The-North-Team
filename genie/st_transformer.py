from torch import nn, Tensor
from einops import rearrange

from genie.attention import SelfAttention, CrossAttention
import torch

class Mlp(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


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
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
    def forward(self, x_TSC: Tensor) -> Tensor:
        # Process attention spatially
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        # Process attention temporally
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)

        # Apply the MLP
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        return x_TSC


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

    def forward(self, tgt):
        x = tgt
        for layer in self.layers:
            x = layer(x)

        return x


class STABlock(nn.Module):
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
        
        self.action_attn = SelfAttention(
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
        
        self.action_crossattn = CrossAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )
        
        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
        self.norm_actions = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp_actions = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
        for name, param in self.spatial_attn.named_parameters():
            param.requires_grad = False
        for name, param in self.temporal_attn.named_parameters():
            param.requires_grad = False
        for name, param in self.norm1.named_parameters():
            param.requires_grad = False   
            
    def forward(self, x_TSC: Tensor, x_TA,) -> Tensor:
        # Process attention spatially
        B, T, S = x_TSC.size(0), x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        # Process attention temporally
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(x_TC, causal=True)

        # Process attention actions
        x_TA = x_TA + self.action_attn(x_TA, causal=True)
        # Apply the MLP on actions
        x_TA = x_TA + self.mlp_actions(self.norm_actions(x_TA))
        
        # Process cross-attention
        x_C = rearrange(x_TC, '(B S) T C -> B (S T) C', B=B, T=T)
        x_C = x_C + self.action_crossattn(x_C, x_TA)
        x_TC = rearrange(x_C, 'B (T S) C -> (B S) T C', B=B, T=T)
        
        # Apply the MLP
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        return x_TSC


class STATransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_action: int,
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
        self.action_encoder = nn.Sequential(
            nn.Linear(d_action, d_model),
            nn.ReLU(),
            Mlp(d_model),
            nn.LayerNorm(d_model, eps=1e-05)
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, d_action),
            nn.ReLU(),
            Mlp(d_action),
            nn.LayerNorm(d_action, eps=1e-05)
        )
        self.future_action_token = nn.Parameter(torch.zeros( 1, d_action))
        self.layers = nn.ModuleList([STABlock(
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

    def forward(self, tgt, act, pos_embed_act):
        x = tgt
        
        B = tgt.size(0)
        future_action_tokens = torch.stack([self.future_action_token]*B).to(act.device)
        act = torch.concatenate(
                (act, future_action_tokens), dim=1)
        act += pos_embed_act
        act = self.action_encoder(act)

        for layer in self.layers:
            x = layer(x, act)
        pred_action = self.action_decoder(act[:, -1])
        return x, pred_action
