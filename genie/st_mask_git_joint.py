import math

import mup
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm
from transformers.utils import ModelOutput

from genie.factorization_utils import FactorizedEmbedding, factorize_labels
from genie.config import GenieConfig
from genie.st_transformer import STTransformerDecoder, STATransformerDecoder

from lam.lam import LatentActionModel

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class ActivationLogger:
    def __init__(self, model, log_dir, log_interval=100):
        self.model = model
        self.log_dir = log_dir
        self.log_interval = log_interval
        # self.activations = defaultdict(list)
        self.mlp_actions_activations = defaultdict(list)
        self.mlp_temporal_activations = defaultdict(list)
        self.cur_step = 0

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir + "/actions", exist_ok=True)
        os.makedirs(log_dir + "/temporal", exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.layers = list(self.model.named_modules())

        self.first_layer_name = 'layers.0'
        self.last_layer_name = 'layers.31'

        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.ReLU, torch.nn.GELU)):
                if name.startswith(self.first_layer_name) or name.startswith(self.last_layer_name):
                    module.register_forward_hook(self.save_activation(name))

    def save_activation(self, name):
        def hook(module, input, output):
            if self.cur_step % self.log_interval == 0:
                # act_array = output.detach().cpu().numpy()
                if 'mlp_actions' in name:
                    self.mlp_actions_activations[name].append(output.detach().cpu().numpy())
                elif 'mlp' in name:
                    self.mlp_temporal_activations[name].append(output.detach().cpu().numpy())

        return hook

    def save_all_activations(self):
        for name, activations in self.mlp_actions_activations.items():
            log_dir=self.log_dir + "/actions"
            act_array = np.concatenate(activations, axis=0)

            flattened_act_array = act_array.flatten()

            plt.figure()
            plt.hist(flattened_act_array, bins=50)
            plt.title(f'{name} Activation Histogram')
            plt.savefig(os.path.join(log_dir, f'{name}_hist.png'))
            plt.close()

            json_data = {name: flattened_act_array.tolist()}
            with open(os.path.join(log_dir, f'{name}_acts.json'), 'w') as f:
                json.dump(json_data, f)
            np.save(os.path.join(log_dir, f'{name}_acts.npy'), act_array)
        
        for name, activations in self.mlp_temporal_activations.items():
            log_dir=self.log_dir + "/temporal"
            act_array = np.concatenate(activations, axis=0)

            flattened_act_array = act_array.flatten()

            plt.figure()
            plt.hist(flattened_act_array, bins=50)
            plt.title(f'{name} Activation Histogram')
            plt.savefig(os.path.join(log_dir, f'{name}_hist.png'))
            plt.close()

            json_data = {name: flattened_act_array.tolist()}
            with open(os.path.join(log_dir, f'{name}_acts.json'), 'w') as f:
                json.dump(json_data, f)
            np.save(os.path.join(log_dir, f'{name}_acts.npy'), act_array)

    def step(self):
        self.cur_step += 1


def cosine_schedule(u):
    """ u in [0, 1] """
    if isinstance(u, torch.Tensor):
        cls = torch
    elif isinstance(u, float):
        cls = math
    else:
        raise NotImplementedError(f"Unexpected {type(u)=} {u=}")

    return cls.cos(u * cls.pi / 2)


class STMaskGIT(nn.Module, PyTorchModelHubMixin):
    # Next-Token prediction as done in https://arxiv.org/pdf/2402.15391.pdf
    def __init__(self, config: GenieConfig, log_activations, activation_log_dir):
        super().__init__()
        self.h = self.w = math.isqrt(config.S)
        assert self.h**2 == config.S, "Expected S to be square"
        
        self.action_conditioned = config.action_conditioned
        self.predict_actions = config.predict_actions
        if self.action_conditioned:
            self.decoder = STATransformerDecoder(
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_model=config.d_model,
                d_action=config.d_action,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                qk_norm=config.qk_norm,
                use_mup=config.use_mup,
                attn_drop=config.attn_drop,
                mlp_ratio=config.mlp_ratio,
                mlp_bias=config.mlp_bias,
                mlp_drop=config.mlp_drop,
            )
            self.action_loss = nn.MSELoss()
            self.lam = LatentActionModel(config)
            
            self.activation_logger = None
            if log_activations:
                self.activation_logger = ActivationLogger(self.decoder, activation_log_dir, log_interval=1)

        else:
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

        # self.pos_embed_TSC = torch.nn.Parameter(torch.zeros(1, config.T, config.S, config.d_model))
        self.pos_embed_TSC = torch.nn.Parameter(torch.zeros(1, config.T, config.S, config.d_model))
        self.mask_token_id = config.image_vocab_size

        self.token_embed = FactorizedEmbedding(  # also works for num_factored_vocabs = 1
            factored_vocab_size=config.factored_vocab_size,
            num_factored_vocabs=config.num_factored_vocabs,
            d_model=config.d_model,
            mask_token_id=self.mask_token_id,
        )

        cls = FixedMuReadout if config.use_mup else nn.Linear  # (Fixed)MuReadout might slow dow down compiled training?
        self.out_x_proj = cls(config.d_model, config.factored_vocab_size * config.num_factored_vocabs)

        self.config = config

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        min_new_tokens: int = None,
        return_logits: int = False,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Args designed to match the format of Llama.
        We ignore `attention_mask`, and use `max_new_tokens` to determine the number of frames to generate.

        Returns: `(sample_THW, factored_logits)` if `return_logits` else `sample_THW`
            sample_THW: size (B, num_new_frames * H * W) corresponding to autoregressively generated
                unfactorized token ids for future frames.
            Optionally, factored_logits: size (B, factored_vocab_size, num_factored_vocabs, num_new_frames, H, W).
        """
        assert min_new_tokens in (None, max_new_tokens), \
            "Expecting `min_new_tokens`, if specified, to match `max_new_tokens`."

        assert max_new_tokens % self.config.S == 0, "Expecting `max_new_tokens` to be a multiple of `self.config.S`."
        num_new_frames = max_new_tokens // self.config.S

        inputs_THW = rearrange(input_ids.clone(), "b (t h w) -> b t h w", h=self.h, w=self.w)
        inputs_masked_THW = torch.cat([
            inputs_THW,
            torch.full((input_ids.size(0), num_new_frames, self.h, self.w),
                       self.mask_token_id, dtype=torch.long, device=input_ids.device)
        ], dim=1)

        all_factored_logits = []
        for timestep in range(inputs_THW.size(1), inputs_THW.size(1) + num_new_frames):
            # could change sampling hparams
            sample_HW, factored_logits = self.maskgit_generate(
                inputs_masked_THW,
                timestep,
                maskgit_steps=maskgit_steps,
                temperature=temperature
            )
            inputs_masked_THW[:, timestep] = sample_HW
            all_factored_logits.append(factored_logits)

        predicted_tokens = rearrange(inputs_masked_THW, "B T H W -> B (T H W)")
        if return_logits:
            return predicted_tokens, torch.stack(all_factored_logits, dim=3)  # (b, factored_vocab_size, num_factored_vocabs, num_new_frames, h, w)
        else:
            return predicted_tokens

    @staticmethod
    def init_mask(prompt_THW):
        # since we generate 1 image at a time, the mask should be for a single frame, not across all frames.
        T, H, W = prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)
        unmasked = torch.zeros(prompt_THW.size(0), H * W, dtype=torch.bool, device=prompt_THW.device)
        return unmasked

    @torch.no_grad()
    def maskgit_generate(
        self,
        prompt_THW: torch.LongTensor,
        out_t: int,
        maskgit_steps: int = 1,
        temperature: float = 0.0,
        unmask_mode: str = "random",
        prompt_TA: torch.LongTensor = None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Performs MaskGIT-style inference to predict frame `out_t`.

        Args:
            prompt_THW: Unfactorized token ids, size (B, T, H, W)
            out_t: Will return predicted unfactorized token ids for this frame.
                Should be >= 1 as the 0th frame is assumed to be given.
                Expects all future frames to be fully masked.
            maskgit_steps: The number of MaskGIT-style inference steps to take.
            temperature: Sampling temperature.
                In the factorized case, sampling is performed for each factorized vocabulary independently.
                If temperature is <= 1e-8, will be greedy (i.e. argmax) instead of actual sampling.
            unmask_mode: The method to determine tokens to unmask during each step of MaskGIT inference.
                Options:
                    - "greedy" for unmasking the most confident tokens, which is matches the original MaskGIT
                    - "random" for randomly choosing tokens to unmask
                "greedy" tends to copy the previous frame, so we default to "random" instead.

        Returns: (sample_HW, factored_logits)
            sample_HW: size (B, H, W) corresponding to predicted unfactorized token ids for frame `out_t`.
            factored_logits: size (B, factored_vocab_size, num_factored_vocabs, H, W).
        """
        # assume we have pre-masked z{out_t}...zT with all masks
        assert out_t, "maskgit_generate requires out_t > 0"
        assert torch.all(prompt_THW[:, out_t:] == self.mask_token_id), \
            f"when generating z{out_t}, frames {out_t} and later must be masked"

        bs, t, h, w = prompt_THW.size(0), prompt_THW.size(1), prompt_THW.size(2), prompt_THW.size(3)

        # this will be modified in place on each iteration of this loop
        unmasked = self.init_mask(prompt_THW)
        
        if prompt_TA is not None:
            logits_CTHW, logits_action = self.compute_logits(prompt_THW, prompt_TA, out_t=out_t)
        else:
            logits_CTHW = self.compute_logits(prompt_THW)

        logits_CHW = logits_CTHW[:, :, out_t]
        orig_logits_CHW = logits_CHW.clone()  # Return these original logits, not logits after partially sampling.
        if prompt_TA is not None:
            logits_action = logits_action[:, out_t]
            orig_logits_action = logits_action.clone()

        for step in tqdm(range(maskgit_steps)):
            # Perform a single maskgit step (cosine schedule), updating unmasked in-place
            if step > 0:  # recompute logits with updated prompt
                if prompt_TA is not None:
                    logits_CHW, logits_action = self.compute_logits(prompt_THW, prompt_TA, out_t=out_t)
                    logits_CHW = logits_CHW[:, :, out_t]
                    logits_action = logits_action[:, out_t]
                else:
                    logits_CHW = self.compute_logits(prompt_THW)[:, :, out_t]

            factored_logits = rearrange(logits_CHW, "b (num_vocabs vocab_size) h w -> b vocab_size num_vocabs h w",
                                        vocab_size=self.config.factored_vocab_size,
                                        num_vocabs=self.config.num_factored_vocabs)

            factored_probs = torch.nn.functional.softmax(factored_logits, dim=1)

            samples_HW = torch.zeros((bs, h, w), dtype=torch.long, device=prompt_THW.device)
            confidences_HW = torch.ones((bs, h, w), dtype=torch.float, device=prompt_THW.device)
            for probs in factored_probs.flip(2).unbind(2):
                if temperature <= 1e-8:  # greedy sampling
                    sample = probs.argmax(dim=1)
                else:
                    # Categorical expects last dim to be channel dim
                    dist = torch.distributions.categorical.Categorical(
                        probs=rearrange(probs, "b vocab_size ... -> b ... vocab_size") / temperature
                    )
                    sample = dist.sample()
                samples_HW *= self.config.factored_vocab_size
                samples_HW += sample
                confidences_HW *= torch.gather(probs, 1, sample.unsqueeze(1)).squeeze(1)

            prev_unmasked = unmasked.clone()
            prev_img_flat = rearrange(prompt_THW[:, out_t], "B H W -> B (H W)")

            samples_flat = samples_HW.reshape(bs, self.config.S)

            if step != maskgit_steps - 1:  # skip masking for last maskgit step
                # use cosine mask scheduling function, n is how many of frame out_t to mask
                n = math.ceil(cosine_schedule((step + 1) / maskgit_steps) * self.config.S)

                if unmask_mode == "greedy":
                    # set the n patches with the least confidence to mask_token
                    confidences_flat = confidences_HW.reshape(bs, self.config.S)
                elif unmask_mode == "random":
                    # randomize confidences, so that patches are randomly masked
                    confidences_flat = torch.rand_like(confidences_HW).reshape(bs, self.config.S)
                    # not probability distribution anymore, but only relative order matters
                else:
                    raise NotImplementedError(f"Expected `unmask_mode` to be one of ['greedy', 'random'], "
                                              f"got {unmask_mode}")

                confidences_flat[unmasked] = torch.inf
                least_confident_tokens = torch.argsort(confidences_flat, dim=1)
                # unmask the (self.config.S - n) most confident tokens
                unmasked.scatter_(1, least_confident_tokens[:, n:], True)
                samples_flat.scatter_(1, least_confident_tokens[:, :n], self.mask_token_id)

            # copy previously unmasked values from prompt input into sample
            samples_flat[prev_unmasked] = prev_img_flat[prev_unmasked]
            samples_HW = samples_flat.reshape(-1, h, w)

            # feed back to iteratively decode
            prompt_THW[:, out_t] = samples_HW
            if prompt_TA is not None:
                prompt_TA[:, out_t] = logits_action 

        # Return the final sample and logits
        return samples_HW, rearrange(
            orig_logits_CHW, "B (num_vocabs vocab_size) H W -> B vocab_size num_vocabs H W",
            vocab_size=self.config.factored_vocab_size, num_vocabs=self.config.num_factored_vocabs, H=h, W=w
        )

    def compute_loss_and_acc(self, logits_CTHW, targets_THW, relevant_mask_THW, weighted=False):

        if weighted:
            orig_targets_THW = targets_THW.clone()

        # Video token prediction
        targets_THW = targets_THW.clone()
        logits_CTHW, targets_THW = logits_CTHW[:, :, 1:], targets_THW[:, 1:]  # first frame always unmasked

        factored_logits = rearrange(logits_CTHW,
                                    "b (num_vocabs vocab_size) t h w -> b vocab_size num_vocabs t h w",
                                    vocab_size=self.config.factored_vocab_size,
                                    num_vocabs=self.config.num_factored_vocabs)

        factored_targets = factorize_labels(targets_THW)

        if weighted:
            orig_factored_targets = factorize_labels(orig_targets_THW)
            c = torch.e
            # using shifted targets for comparing to the next frame
            token_changes = (orig_factored_targets[:, :, 1:, ...] != orig_factored_targets[:, :, :-1, ...]).float()
            motion_weights = c ** token_changes
            loss_THW = F.cross_entropy(factored_logits, factored_targets, reduction="none") * motion_weights
            loss_THW = loss_THW.sum(dim=1)
        else:
            loss_THW = F.cross_entropy(factored_logits, factored_targets, reduction="none").sum(dim=1)

        acc_THW = (factored_logits.argmax(dim=1) == factored_targets).all(dim=1)

        # Compute the mean masked error.
        # Multiply loss values by mask instead of indexing them, more computationally efficient.
        num_masked_tokens = torch.sum(relevant_mask_THW)
        relevant_loss = torch.sum(loss_THW * relevant_mask_THW) / num_masked_tokens
        relevant_acc = torch.sum(acc_THW * relevant_mask_THW).float() / num_masked_tokens

        # only optimize on the masked/noised logits?
        return relevant_loss, relevant_acc

    def compute_logits(self, x_THW, x_TA=None, out_t=None):
        # x_THW is for z0,...,zT while x_targets is z1,...,zT
        x_TS = rearrange(x_THW, "B T H W -> B T (H W)")
        x_TSC = self.token_embed(x_TS)

        if self.action_conditioned:
            if not out_t:
                x_TSC = self.decoder(x_TSC + self.pos_embed_TSC, x_TA)
            else:
                # TODO: generation/eval
                pass
        else:
            x_TSC = self.decoder(x_TSC + self.pos_embed_TSC)

        x_next_TSC = self.out_x_proj(x_TSC)

        logits_CTHW = rearrange(x_next_TSC, "B T (H W) C -> B C T H W", H=self.h, W=self.w)

        if self.action_conditioned:
            return logits_CTHW, x_TA
        else:
            return logits_CTHW

    def forward(self, input_ids, labels, actions=None, labels_actions=None):
        T, H, W = self.config.T, self.h, self.w
        x_THW = rearrange(input_ids, "B (T H W) -> B T H W", T=T, H=H, W=W)

        if self.action_conditioned:
            # logits_CTHW = self.compute_logits(x_THW, actions)
            pred_actions_output = self.lam(input_ids, labels, actions, labels_actions)
            pred_actions_logits = pred_actions_output.logits
            
            # the actions fed into the video decoder should be the first gt action (which is free) and the remaining 15 predicted
            action_input = torch.cat([pred_actions_output.encoded_actions[:, 0], pred_actions_logits[:, :-1]], dim=1)
            
            logits_CTHW = self.compute_logits(input_ids, action_input)

            if self.predict_actions:
                action_loss = pred_actions_output.loss

        else:
            logits_CTHW = self.compute_logits(x_THW)

        labels = rearrange(labels, "B (T H W) -> B T H W", T=T, H=H, W=W)

        # Record the loss over masked tokens only to make it more comparable to LLM baselines
        relevant_mask = x_THW[:, 1:] == self.mask_token_id  # could also get mask of corrupted tokens by uncommenting line in `get_maskgit_collator`
        relevant_loss, relevant_acc = self.compute_loss_and_acc(logits_CTHW, labels, relevant_mask)

        if self.action_conditioned and self.predict_actions:
            relevant_loss += action_loss
        
        if self.activation_logger:
            self.activation_logger.step()

        return ModelOutput(loss=relevant_loss, acc=relevant_acc, logits=logits_CTHW)

    def init_weights(self):
        """ Works with and without muP. """
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module.weight, "infshape"):  # muP
                    mup.normal_(module.weight, mean=0.0, std=std)
                else:
                    module.weight.data.normal_(mean=0.0, std=std)

                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def set_mup_shapes(self, rescale_params=False, log_activations=False, activation_log_dir=None):
        base_config = self.config.shallow_copy()
        base_config.num_heads = 8
        base_config.d_model = 256  # currently hardcoding to this shape
        base_model = STMaskGIT(base_config, log_activations, activation_log_dir)

        mup.set_base_shapes(self, base_model, rescale_params=rescale_params)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """ Extra logic for muP. """
        model = super().from_pretrained(*args, **kwargs)
        if model.config.use_mup:
            model.set_mup_shapes(rescale_params=False, **kwargs)

        return model


class FixedMuReadout(mup.MuReadout):
    def forward(self, x):
        """
        Using `return super(mup.MuReadout, self).forward(self.output_mult * x / self.width_mult())` with `torch.compile`
        results in two divisions by `self.width_mult()` for some reason
        """
        # return F.linear(self.output_mult * x / self.width_mult(), self.weight, self.bias)  # equivalent
        return nn.Linear.forward(self, self.output_mult * x / self.width_mult())
