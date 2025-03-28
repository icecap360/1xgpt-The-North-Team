from torch import nn, Tensor
from einops import rearrange

from genie.attention import SelfAttention, CrossAttention
import torch

class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

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
        # self.act = SwiGLU(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

# def modulate(x, shift, scale):
#     return x * (1 + scale) + shift
    
# class ModulateLayer(nn.Module):
#     """
#     Modified from the final layer adopted from DiT with token-wise modulation.
#     """

#     def __init__(self, model_channels, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)

#         self.linear_out = nn.Linear(out_channels, out_channels, bias=True)
#         self.adaLN_modulation = nn.Sequential(
#             nn.Linear(model_channels, model_channels), nn.SiLU(), nn.Linear(model_channels, 3 * out_channels, bias=True)
#         )
#         self.apply(self._init_weights)

#     def forward(self, x, c):
#         """
#         a simple modulation
#         """
#         x_shape = x.shape
#         x = rearrange(x, "(b s) t d -> b s t d", b=len(c))
#         c = c[:, None, : x_shape[2]]
#         shift, scale = self.adaLN_modulation(c).chunk(3, dim=-1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear_out(x)
#         return x.view(x_shape)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#             if m.weight is not None:
#                 nn.init.constant_(m.weight, 1.0)

def modulate(x, shift, scale, temporal=True):
    x_shape = x.shape
    if temporal:
        x = rearrange(x, "(b s) t d -> b s t d", b=len(scale))
    else:
        x = rearrange(x, "(b t) s d -> b t s d", b=len(scale))
    
    x = x * (1 + scale) + shift
    return x.view(x_shape)

def scale(x, gate, temporal=True):
    x_shape = x.shape
    if temporal:
        x = rearrange(x, "(b s) t d -> b s t d", b=len(gate))
    else:
        x = rearrange(x, "(b t) s d -> b t s d", b=len(gate))
    
    x = x * gate
    return x.view(x_shape)

class ModulateLayer(nn.Module):
    """
    Modified from the final layer adopted from DiT with token-wise modulation.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)

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

        c_t = c[:, None, :, :] # B, S, T, C
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
            is_2d=True
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
            is_2d=False,
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
        # sequence dim is over each frame's 16x16 patch tokens
        self.spatial_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
            is_2d=True,
            seq_len=256
        )

        # self.spatial_attn_2 = SelfAttention(
        #     num_heads=num_heads,
        #     d_model=d_model,
        #     qkv_bias=qkv_bias,
        #     proj_bias=proj_bias,
        #     qk_norm=qk_norm,
        #     use_mup=use_mup,
        #     attn_drop=attn_drop,
        #     is_2d=True,
        #     rope=False
        # )
        
        # self.action_attn = SelfAttention(
        #     num_heads=num_heads,
        #     d_model=d_model,
        #     qkv_bias=qkv_bias,
        #     proj_bias=proj_bias,
        #     qk_norm=qk_norm,
        #     use_mup=use_mup,
        #     attn_drop=attn_drop,
        # )

        # sequence dim is over time sequence (16)
        self.temporal_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
            is_2d=False,
            seq_len=16
        )

        self.modulate_actions = ModulateLayer(d_model,d_model)
        # self.modulate_video = ModulateLayer(d_model,d_model)
        
        # self.norm2 = nn.Identity() if qk_norm else RMSNorm(d_model, eps=1e-05)
        # self.norm_temporal = nn.Identity() if qk_norm else RMSNorm(d_model, eps=1e-05)
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        # self.mlp_actions = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        # self.mlp_slots = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
        # self.norm_actions = nn.Identity() if qk_norm else RMSNorm(d_model, eps=1e-05)
        # self.norm_actions_2 = nn.Identity() if qk_norm else RMSNorm(d_model, eps=1e-05)
        # self.norm_actions_3 = nn.Identity() if qk_norm else RMSNorm(d_model, eps=1e-05)
        # self.mlp_actions = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)

        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # self.norm3 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.norm4 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)

        # self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-05)
        # self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-05)
        self.norm3 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-05)
        # self.norm4 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-05)
        
        # commented for rope
        # for name, param in self.spatial_attn.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.temporal_attn.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.norm1.named_parameters():
        #     param.requires_grad = False   
    
    # def forward(self, x_TSC: Tensor, x_TA) -> Tensor:
    #     B, T, S = x_TSC.size(0), x_TSC.size(1), x_TSC.size(2)

    #     scale_t, shift_t, scale_s, shift_s, scale_mlp, shift_mlp, scale_t_2, scale_s_2, scale_mlp_2 = self.modulate_actions(x_TA)
    #     # scale_t, shift_t, scale_mlp, shift_mlp, scale_t_2, scale_mlp_2 = self.modulate_actions(x_TA)

    #     x_SC = rearrange(x_TSC, "B T S C -> (B T) S C")
    #     x_SC = modulate(self.norm1(x_SC), shift_s, scale_s, temporal=False)
    #     x_SC = x_SC + modulate(self.spatial_attn(x_SC), 0, scale_s_2, temporal=False)
    #     # x_SC = rearrange(x_TSC, "B T S C -> (B T) S C")
    #     # x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

    #     # x_TA = x_TA + self.action_attn(x_TA, causal=True)
    #     # # Apply the MLP on actions
    #     # x_TA = x_TA + self.mlp_actions(self.norm_actions(x_TA))

    #     # Process attention temporally
    #     x_TC = rearrange(x_SC, "(B T) S C -> (B S) T C", T=T)
    #     x_TC = modulate(self.norm3(x_TC), shift_t, scale_t)
    #     x_TC = x_TC + modulate(self.temporal_attn(x_TC, causal=True), 0, scale_t_2)

    #     # x_TSC = rearrange(x_TC, "(B S) T C -> B T S C", S=S)
    #     # x_video = rearrange(x_TSC[:, :, :256, :], "B T S C -> (B S) T C")
    #     # x_act = rearrange(x_TSC[:, :, 256:, :], "B T S C -> (B S) T C")
    #     # x_video = x_video + self.modulate_video(self.norm4(x_video), x_TA)
    #     # x_video = rearrange(x_video, "(B S) T C -> B T S C", B=B)
    #     # x_act = x_act + self.modulate_actions(self.norm3(x_act), x_TA)
    #     # x_act = rearrange(x_act, "(B S) T C -> B T S C", B=B)
    #     # x_TSC = torch.concat((x_video, x_act), dim=2)  # [B, T, S + 64, D]
    #     # x_TC = rearrange(x_TSC, "B T S C -> (B S) T C", T=T)

    #     x_TC = modulate(self.norm2(x_TC), shift_mlp, scale_mlp)
    #     x_TC = x_TC + modulate(self.mlp(self.norm2(x_TC)), 0, scale_mlp_2)

    #     x_TSC = rearrange(x_TC, "(B S) T C -> B T S C", S=S)
    #     return x_TSC

    def forward(self, x_TSC: Tensor, x_TA) -> Tensor:
        B, T, S = x_TSC.size(0), x_TSC.size(1), x_TSC.size(2)

        scale_t, shift_t, gate_t = self.modulate_actions(x_TA)

        x_SC = rearrange(x_TSC, "B T S C -> (B T) S C")
        x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        # x_TA = x_TA + self.action_attn(x_TA, causal=True)
        # # Apply the MLP on actions
        # x_TA = x_TA + self.mlp_actions(self.norm_actions(x_TA))

        # Process attention temporally
        # x_TC = rearrange(x_SC, "(B T) S C -> (B S) T C", T=T)
        # x_TC = x_TC + self.modulate_actions(x_TC, x_TA)

        # x_TC = x_TC + self.temporal_attn(x_TC, causal=True)
        # x_TC = x_TC + self.mlp(self.norm2(x_TC))

        x_TC = rearrange(x_SC, "(B T) S C -> (B S) T C", T=T)
        x_TC = modulate(self.norm3(x_TC), shift_t, scale_t)
        x_TC = x_TC + scale(self.temporal_attn(x_TC, causal=True), gate_t)

        x_TC = x_TC + self.mlp(self.norm2(x_TC))

        x_TSC = rearrange(x_TC, "(B S) T C -> B T S C", S=S)
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
        # self.action_encoder = nn.Sequential(
        #     nn.Linear(d_action, d_model),
        #     nn.ReLU(),
        #     Mlp(d_model),
        #     nn.LayerNorm(d_model, eps=1e-05)
        # )

        # self.action_decoder = nn.Sequential(
        #     nn.Linear(d_model, d_action),
        #     nn.ReLU(),
        #     Mlp(d_action),
        #     nn.LayerNorm(d_action, eps=1e-05)
        # )

        self.future_action_token = nn.Parameter(torch.zeros( 1, d_action))
        # self.additive_action_embedding = nn.Parameter(torch.zeros(16, d_model))

        self.slot_embd = torch.nn.Parameter(torch.zeros(1, d_model))
        # self.pos_spatial_embedding = nn.Parameter(torch.zeros(1, d_model, d_action))

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

    def forward(self, tgt, act):
        x = tgt
        
        B = tgt.size(0)
        # act += pos_embed_act
        
        for layer in self.layers:
            x = layer(x, act)

        # act = self.action_decoder(act)
        return x