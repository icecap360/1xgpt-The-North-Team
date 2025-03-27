import sys

sys.path.append("../genie")
import torch.nn as nn
import torch
from einops import rearrange
from transformers.utils import ModelOutput

from genie.factorization_utils import FactorizedEmbedding
from genie.st_transformer import STBlock, Mlp
from genie.attention import SelfAttention, CrossAttention
from huggingface_hub import PyTorchModelHubMixin

# from stvivit import STViViT
# from decode_latents_utils import decode_latents_wrapper

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
            rope=False
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
            rope=False,
        )
        
        self.video_crossattn = CrossAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
            d_kv=d_model * 256
        )
        
        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.norm3 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
        self.norm_video = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # self.mlp_actions = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        
        # for name, param in self.spatial_attn.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.temporal_attn.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.norm1.named_parameters():
        #     param.requires_grad = False   
            
    def forward(self, x_TA, x_TSC):
        # Process attention spatially
        B, T, C = x_TA.size(0), x_TA.size(1), x_TA.size(2)
        # x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        # x_SC = x_SC + self.spatial_attn(self.norm1(x_SC))

        # Process attention temporally
        x_TA = x_TA + self.temporal_attn(self.norm1(x_TA), causal=True)

        norm_x_TSC = rearrange(self.norm_video(x_TSC), "B T S C -> B T (S C)")
        x_TA = x_TA + self.video_crossattn(self.norm3(x_TA), norm_x_TSC)

        x_TA = x_TA + self.mlp(self.norm2(x_TA))

        return x_TA

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

    def forward(self, x_TA, x_TSC):
        x = x_TA
        for layer in self.layers:
            x = layer(x, x_TSC)

        return x

class LatentActionModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.decoder = STTransformerDecoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            qk_norm=config.qk_norm,
            use_mup=config.use_mup,
            attn_drop=config.attn_drop,
            mlp_ratio=config.mlp_ratio,
            mlp_bias=config.mlp_bias,
            mlp_drop=config.mlp_drop,
        )
        self.action_proj = nn.Linear(config.d_model, config.d_model)
        self.action_decoder = nn.Sequential(
                nn.Linear(config.d_model, config.d_action),
                nn.ReLU(),
                Mlp(config.d_action),
                nn.LayerNorm(config.d_action, eps=1e-05)
            )
        self.action_encoder = nn.Sequential(
                nn.Linear(config.d_action, config.d_model),
                nn.ReLU(),
                Mlp(config.d_model),
                nn.LayerNorm(config.d_model, eps=1e-05)
            )
        self.norm = nn.LayerNorm(config.d_model)

        self.action_loss = nn.MSELoss()
        self.config = config

        self.mask_token_id = config.image_vocab_size

        self.token_embed = FactorizedEmbedding(  # also works for num_factored_vocabs = 1
            factored_vocab_size=config.factored_vocab_size,
            num_factored_vocabs=config.num_factored_vocabs,
            d_model=config.d_model,
            mask_token_id=self.mask_token_id,
        )
        self.pos_embed_TSC = torch.nn.Parameter(torch.zeros(1, config.T-1, config.S, config.d_model))
        self.pos_embed_TA = torch.nn.Parameter(torch.zeros(1, config.T-1, config.d_model))

        self.apply(init_weights)
    
    def compute_actions(self, x_TA, x_TSC):
        decoded_actions = self.decoder(x_TA, x_TSC)
        decoded_actions = self.action_proj(decoded_actions) # B, T, C
        return self.norm(decoded_actions)
    
    def forward(self, input_ids, labels, actions=None, labels_actions=None):
        T, H, W = self.config.T, int(self.config.S ** 0.5), int(self.config.S ** 0.5)

        # use labels since input_ids are masked
        x_THW = rearrange(labels, "B (T H W) -> B T H W", T=T, H=H, W=W)
        x_TS = rearrange(x_THW, "B T H W -> B T (H W)", T=T, H=H, W=W)
        x_TSC = self.token_embed(x_TS)[:, :-1]

        x_TA = self.action_encoder(actions)

        pred_actions_logits = self.compute_actions(x_TA + self.pos_embed_TA, x_TSC + self.pos_embed_TSC)
        pred_actions = self.action_decoder(pred_actions_logits)

        loss = self.action_loss(pred_actions, labels_actions)

        return ModelOutput(loss=loss, logits=pred_actions, encoded_actions=x_TA, acc=1.0)
    
    # @classmethod
    # def from_pretrained(cls, *args, **kwargs):
    #     """ Extra logic for muP. """
    #     model = super().from_pretrained(config)
    #     if model.config.use_mup:
    #         model.set_mup_shapes(rescale_params=False, **kwargs)

    #     return model
    
def init_weights(self):
    """ Works with and without muP. """
    std = 0.02
    for module in self.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# class LatentActionModel(nn.Module, PyTorchModelHubMixin):
#     def __init__(
#         self,
#         config,
#     ):
#         super().__init__()

#         self.vivit = STViViT(config)
#         self.apply(init_weights)
    
#     def compute_actions(self, x):
#         # expecting x to have shape [B, T, S, C]
#         encoded_x = self.encoder(x)
#         actions = self.action_proj(rearrange(encoded_x, "B T S C -> B T (S C)"))
#         return self.norm(actions)
    
#     def forward(self, input_ids, labels, actions=None, labels_actions=None):
#         T, H, W = self.config.T, int(self.config.S ** 0.5), int(self.config.S ** 0.5)

#         x_THW = rearrange(input_ids, "B (T H W) -> B T H W", T=T, H=H, W=W)
#         x_TS = rearrange(x_THW, "B T H W -> B T (H W)", T=T, H=H, W=W)
#         x_TSC = self.token_embed(x_TS)

#         pred_actions_logits = self.compute_actions(x_TSC + self.pos_embed_TSC)
#         pred_actions = self.action_decoder(pred_actions_logits)

#         loss = self.action_loss(pred_actions, labels_actions)

#         return ModelOutput(loss=loss, logits=pred_actions)
    
# def init_weights(self):
#     """ Works with and without muP. """
#     std = 0.02
#     for module in self.modules():
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)

#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()