from torch import nn, Tensor
from einops import rearrange

from genie.attention import SelfAttention
from genie.rope import RotaryPositionEmbeddingPytorchV2
from genie.fft2d import FFT2D, InverseFFT2D
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
    
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
    
class ModulateLayer(nn.Module):
    """
    Modified from the final layer adopted from DiT with token-wise modulation.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()

        # self.linear_out = nn.Linear(out_channels, out_channels, bias=True)
        # self.adaLN_modulation_spatial = nn.Sequential(
        #     nn.Linear(model_channels, model_channels), nn.SiLU(), nn.Linear(model_channels, 3 * out_channels, bias=True)
        # )
        self.adaLN_modulation_temporal = nn.Sequential(
            nn.Linear(model_channels, model_channels), nn.SiLU(), nn.Linear(model_channels, 3 * out_channels, bias=True)
        )
        self.d_model = model_channels
        self.apply(self._init_weights)

    def forward(self, c):
        """
        regress modulation parameters
        """
        # c_s = c[:, :, None, :] # B, T, S, C
        # scale_s, shift_s, scale_s_2 = self.adaLN_modulation_spatial(c_s).chunk(3, dim=-1)

        c_first = c[:, 0:1, :]
        c_second = c[:, 1:9, :].mean(dim=1, keepdim=True)
        c_third = c[:, 9:17, :].mean(dim=1, keepdim=True)

        c_pooled = torch.cat([c_first, c_second, c_third], dim=1) # B, 3, C
        c_t = c_pooled[:, None, :, :]  # (B, S=1, T=3, C)
        scale_t, shift_t, gate_t = self.adaLN_modulation_temporal(c_t).chunk(3, dim=-1)
        
        # return scale_t, shift_t, scale_mlp, shift_mlp, scale_t_2, scale_mlp_2
        return scale_t, shift_t, gate_t

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.constant_(m.weight, 0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

def pool_actions_for_tokens(x_actions: Tensor) -> Tensor:
    # x_actions: [B, 33, action_dim]
    token_to_frames = {
        0: [0],
        1: list(range(1, 9)),
        2: list(range(9, 17)),
        3: [17],
        4: list(range(18, 26)),
        5: list(range(26, 34)),
    }
    pooled = [x_actions[:, frames].mean(dim=1) for frames in token_to_frames.values()]
    return torch.stack(pooled, dim=1)  # [B, 6, action_dim]

def to_spatial(x, S=1025):
    if len(x.shape) == 4:
        return rearrange(x, 'B T S C -> (B T) S C')
    else:
        return rearrange(x, '(B S) T C -> (B T) S C', S=S)

def to_temporal(x, T=6):
    if len(x.shape) == 4:
        return rearrange(x, 'B T S C -> (B S) T C')
    else:
        return rearrange(x, '(B T) S C -> (B S) T C', T=T)

# # regular
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
        use_rope = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # sequence dim is over each frame's 16x16 patch tokens
        
        self.spatial_rope = None
        self.temporal_rope = None
        if use_rope:
            self.head_dim = d_model // num_heads
            self.spatial_rope_config = self._create_2d_rope_config()
            
            self.spatial_rope = RotaryPositionEmbeddingPytorchV2(
                seq_len=32*32, training_type=None, **self.spatial_rope_config
            )

            self.temporal_rope_config = self._create_1d_rope_config()
            
            self.temporal_rope = RotaryPositionEmbeddingPytorchV2(
                seq_len=3, training_type=None, **self.temporal_rope_config
            )

        self.spatial_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
            rope=self.spatial_rope,
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
            rope=self.temporal_rope
        )

        # self.modulate_actions = ModulateLayer(d_model,d_model)
        
        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.norm3 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=False)
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.action_mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
    def forward(self, x_TSC: Tensor, x_TA=None) -> Tensor:

        # Process attention spatially
        # scale_t, shift_t, gate_t = self.modulate_actions(x_TA)

        T, S = x_TSC.size(1), x_TSC.size(2)
        # T, S = x_TSC.size(1), x_TSC.size(2)
        # x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        # modulate
        # x_STC_orig = rearrange(x_SC, '(B T) S C -> B S T C', T=T)
        # x_STC = modulate(self.norm3(x_STC_orig), shift_t, scale_t)

        # attn
        # x_TC = rearrange(x_STC, 'B S T C -> (B S) T C', T=T)
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        x_TC = x_TC + self.temporal_attn(self.norm3(x_TC), causal=True)
        # x_STC = rearrange(x_TC, '(B S) T C -> B S T C', T=T, S=S)

        # gate
        # x_STC = x_STC_orig + x_STC * gate_t
        # x_TC = rearrange(x_STC, 'B S T C -> (B S) T C', T=T)

        # Apply the MLP
        x_TC = x_TC + self.mlp(self.norm2(x_TC))
        x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        return x_TSC
    
    def _create_2d_rope_config(self):
        return {
            "dim": self.head_dim,
            "rope_theta": 10000.0,
            "rope_dim": "2D",
            "latent_shape": (32, 32),
        }

    def _create_1d_rope_config(self):
        return {
            "dim": self.head_dim,
            "rope_theta": 10000.0,
            "max_position_embeddings" : 6,
            "rope_dim": "1D",
            "latent_shape": (6),
        }

# class STBlock(nn.Module):
#     # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
#     def __init__(
#         self,
#         num_heads: int,
#         d_model: int,
#         qkv_bias: bool = False,
#         proj_bias: bool = True,
#         qk_norm: bool = True,
#         use_mup: bool = True,
#         attn_drop: float = 0.0,
#         mlp_ratio: float = 4.0,
#         mlp_bias: bool = True,
#         mlp_drop: float = 0.0,
#         use_rope = False
#     ) -> None:
#         super().__init__()
#         self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
#         # sequence dim is over each frame's 16x16 patch tokens
        
#         self.spatial_rope = None
#         self.temporal_rope = None
#         if use_rope:
#             self.head_dim = d_model // num_heads
#             self.spatial_rope_config = self._create_2d_rope_config()
            
#             self.spatial_rope = RotaryPositionEmbeddingPytorchV2(
#                 seq_len=32*32, training_type=None, **self.spatial_rope_config
#             )

#             self.temporal_rope_config = self._create_1d_rope_config()
            
#             self.temporal_rope = RotaryPositionEmbeddingPytorchV2(
#                 seq_len=3, training_type=None, **self.temporal_rope_config
#             )

#         self.spatial_attn = SelfAttention(
#             num_heads=num_heads,
#             d_model=d_model,
#             qkv_bias=qkv_bias,
#             proj_bias=proj_bias,
#             qk_norm=qk_norm,
#             use_mup=use_mup,
#             attn_drop=attn_drop,
#             rope=self.spatial_rope,
#         )

#         # sequence dim is over time sequence (16)
#         self.temporal_attn = SelfAttention(
#             num_heads=num_heads,
#             d_model=d_model,
#             qkv_bias=qkv_bias,
#             proj_bias=proj_bias,
#             qk_norm=qk_norm,
#             use_mup=use_mup,
#             attn_drop=attn_drop,
#             rope=self.temporal_rope
#         )

#         # self.modulate_actions = ModulateLayer(d_model,d_model)
        
#         self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
#         self.norm3 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05, elementwise_affine=False)
#         self.norm4 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
#         self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
#         self.action_mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
#     def forward(self, x_TSC: Tensor, x_TA=None) -> Tensor:
#         # x_TSC: [B, T, S, C] — video patch tokens
#         # x_TA:  [B, T, 1, C] — action tokens

#         B, T, S, C = x_TSC.shape

#         T, S = x_TSC.size(1), x_TSC.size(2)
#         x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
#         x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))
#         x_TSC = rearrange(x_SC, '(B T) S C -> B T S C', T=T)

#         # -------- Temporal Attention --------
#         x_temporal_vid = to_temporal(x_TSC)  # [(B*S), T, C] 
#         x_temporal_act = to_temporal(x_TA)   # [B, T, C]
#         x_temporal_act = x_temporal_act.unsqueeze(1).repeat(1, S, 1, 1)  # [B, S, T, C]
#         x_temporal_act = rearrange(x_temporal_act, 'B S T C -> (B S) T C')  # [(B*S), T, C]

#         x_temporal_concat = torch.cat([x_temporal_vid, x_temporal_act], dim=1)  # [(B*S), 2T, C]
#         temporal_norm = self.norm3(x_temporal_concat)

#         temporal_attn_out = self.temporal_attn(temporal_norm, causal=True)  # [(B*S), 2T, C]

#         x_temporal_vid_out = temporal_attn_out[:, :T, :]   # [(B*S), T, C]
#         x_temporal_act_out = temporal_attn_out[:, T:, :]   # [(B*S), T, C]

#         # Residual after temporal attention
#         x_temporal_vid = x_temporal_vid + x_temporal_vid_out
#         x_temporal_act = x_temporal_act + x_temporal_act_out

#         # -------- MLPs --------
#         x_temporal_vid = x_temporal_vid + self.mlp(self.norm2(x_temporal_vid))  # [(B*S), T, C]
#         x_temporal_act = x_temporal_act + self.action_mlp(self.norm4(x_temporal_act))  # [(B*S), T, C]

#         # -------- Reshape back to original format --------
#         x_TSC = rearrange(x_temporal_vid, '(B S) T C -> B T S C', B=B, S=S)
#         x_temporal_act = rearrange(x_temporal_act, '(B S) T C -> B S T C', B=B, S=S)
#         x_TA = x_temporal_act.mean(dim=1)  # shape: (B, T, C)
#         x_TA = rearrange(x_TA, 'B T C -> B T 1 C')  # match shape: (B, T, 1, C)

#         return x_TSC, x_TA
    
#     def _create_2d_rope_config(self):
#         return {
#             "dim": self.head_dim,
#             "rope_theta": 10000.0,
#             "rope_dim": "2D",
#             "latent_shape": (32, 32),
#         }

#     def _create_1d_rope_config(self):
#         return {
#             "dim": self.head_dim,
#             "rope_theta": 10000.0,
#             "max_position_embeddings" : 12,
#             "rope_dim": "1D",
#             "latent_shape": (12),
#         }


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
        use_rope=True,
        with_act=True
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
            use_rope=use_rope
        ) for _ in range(num_layers)])
        self.with_act = with_act

    def forward(self, tgt, act=None):
        x = tgt

        if self.with_act:
            act = pool_actions_for_tokens(act).unsqueeze(2) # B, T, 1, C
            for layer in self.layers:
                x, act = layer(x, act)
        else:
            for layer in self.layers:
                x = layer(x)

        return x
