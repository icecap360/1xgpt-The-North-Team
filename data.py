import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset

from genie.factorization_utils import factorize_token_ids, unfactorize_token_ids
from genie.config import GenieConfig
from genie.st_mask_git import cosine_schedule


class RawTokenDataset(TorchDataset):
    """ Loads raw uint32 tokens as memmap-backed array """
    def __init__(
        self,
        data_dir,
        is_eval = False,
        window_size=17,
        stride=1,
        filter_overlaps=False
    ):
        """
        Args:
            data_dir: directory with the same format as `/home/aditya/1xgpt-The-North-Team_testing/data/train_v2.0`.
                Notably, has `metadata.json`, `metadata/metadata_{rank}.json`, `videos/video_{rank}.bin`, and `segment_indices/segment_idx_{rank}.bin`.
            window_size: number of frames per "video" sequence (default is 17)
            stride: frame skip (default is 1)
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g., frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
        """
        self.data_dir = Path(data_dir)
        self.metadata = json.load(open(self.data_dir / "metadata.json"))
        
        # Determine the number of ranks from metadata.json
        self.num_ranks = self.metadata["num_shards"]
        
        # Load metadata for each rank
        self.metadata_shards = []

        for rank in range(self.num_ranks):
            if is_eval:
                self.metadata_shards.append(json.load(open(self.data_dir / f"metadata_{rank}.json")))
            else:
                self.metadata_shards.append(json.load(open(self.data_dir / f"metadata/metadata_{rank}.json")))

        self.window_size = window_size
        self.stride = stride
        self.filter_overlaps = filter_overlaps

        # Initialize data and segment indices for each rank
        self.data_list = []
        self.segment_idx_list = []
        for rank in range(self.num_ranks):
            total_frames = self.metadata_shards[rank]["shard_num_frames"]

            if is_eval:
                filename = self.data_dir / f"video_{rank}.bin"
            else:
                filename = self.data_dir / f"videos/video_{rank}.bin"
            # Assuming each frame is encoded into a 3x32x32 tensor
            self.data_list.append(np.memmap(filename, dtype=np.int32, mode="r", shape=(math.ceil(total_frames / 17), 3, 32, 32)))
            
            # filename = self.data_dir / f"segment_indices/segment_idx_{rank}.bin"
            # self.segment_idx_list.append(np.memmap(filename, dtype=np.int32, mode="r", shape=(total_frames)))
        self.data_list = np.concatenate(self.data_list, axis=0)
        self.valid_start_inds = [i for i in range(len(self.data_list))]

        # # Generate valid start indices for each rank
        # self.valid_start_inds_list = []
        # for rank in range(self.num_ranks):
        #     total_frames = self.metadata_shards[rank]["shard_num_frames"]
        #     valid_start_inds = []
        #     for start_ind in range(0, total_frames - self.window_size + 1, self.stride):
        #         valid_start_inds.append(start_ind)
            
        #     if self.filter_overlaps:
        #         filtered_start_inds = []
        #         for start_ind in valid_start_inds:
        #             overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
        #             for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
        #                 if existing_start_ind in overlapping_start_inds:
        #                     break
        #             else:
        #                 filtered_start_inds.append(start_ind)
        #         valid_start_inds = filtered_start_inds
            
        #     self.valid_start_inds_list.append(valid_start_inds)

    def __len__(self):
        total_len = len(self.valid_start_inds)
        return total_len

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        idx = self.valid_start_inds[idx]
            
        data = self.data_list[idx]
        
        # Flatten the sequence
        x = torch.from_numpy(data.astype(np.int32))
        x = x.flatten()

        attention_mask = torch.ones_like(x)
        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
        }


def get_maskgit_collator(config: GenieConfig):
    mask_token_id = config.image_vocab_size
    h = w = math.isqrt(config.S)

    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1

        input_ids = torch.stack([ex["input_ids"] for ex in features])
        device = input_ids.device
        x_THW = rearrange(input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T,
                          h=h, w=w)
        x_THWC = factorize_token_ids(x_THW, config.num_factored_vocabs, config.factored_vocab_size)
        labels = x_THW.to(torch.int64).clone()

        # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
        r = torch.rand(x_THWC.size(), device=device)
        u01 = torch.rand((), device=device)
        random_patches_mask = r < config.max_corrupt_rate * u01
        random_values = torch.randint(low=0, high=config.factored_vocab_size, size=x_THWC.size(),
                                      dtype=torch.long, device=device)
        x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(0.25, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand((len(features), h, w, config.num_factored_vocabs), device=device)
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[:, first_masked_frame + i][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = config.num_prompt_frames

        mask = torch.zeros(1)
        c = 0
        while mask.max() == 0:  # We could get unlucky and mask no tokens?
            # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
            mask_prob_T = cosine_schedule(torch.rand(len(features), config.T - first_masked_frame, 1, 1))

            r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob_T
            c += 1

        if c > 1:
            print(f"Generated mask {c} > 1 times.")

        x_THW = unfactorize_token_ids(x_THWC, config.num_factored_vocabs, config.factored_vocab_size)
        x_THW[:, first_masked_frame:][mask] = mask_token_id

        return {
            "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
        }

    return collate_fn

# def get_maskgit_collator(config: GenieConfig):
#     mask_token_id = config.image_vocab_size
#     h = w = math.isqrt(config.S)

#     def collate_fn(features) -> dict[str, torch.Tensor]:
#         # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
#         # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1

#         input_ids = torch.stack([ex["input_ids"] for ex in features])
#         device = input_ids.device
#         x_THW = rearrange(input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T,
#                           h=h, w=w)
#         # x_THWC = factorize_token_ids(x_THW, config.num_factored_vocabs, config.factored_vocab_size)
#         x_THW = x_THW.to(torch.int64)

#         labels = x_THW.clone()

#         # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
#         r = torch.rand(x_THW.size(), device=device)
#         u01 = torch.rand((), device=device)
#         random_patches_mask = r < config.max_corrupt_rate * u01
#         random_values = torch.randint(low=0, high=config.factored_vocab_size, size=x_THW.size(),
#                                       dtype=torch.int64, device=device)
#         x_THW[random_patches_mask] = random_values[random_patches_mask]

#         if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
#             # Leave frames [0, first_masked_frame) unmasked.
#             first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
#             x_THWC_view = x_THW[:, first_masked_frame:]

#             # Arbitrary numbers here, but corrupting later frames more
#             # since we likely have compounding errors.
#             correct_rate = random.uniform(0.25, 1.0)
#             for i in range(x_THWC_view.size(1)):
#                 correct_rate *= random.uniform(0.9, 1.0)
#                 r = torch.rand((len(features), h, w), device=device)
#                 random_patches_mask = r > correct_rate
#                 x_THWC_view[:, i][random_patches_mask] = random_values[:, first_masked_frame + i][random_patches_mask]
#         else:  # Typical MLM masking
#             first_masked_frame = 1

#         mask = torch.zeros(1)
#         c = 0
#         while mask.max() == 0:  # We could get unlucky and mask no tokens?
#             # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
#             mask_prob_T = cosine_schedule(torch.rand(len(features), config.T - first_masked_frame, 1, 1))

#             r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
#             mask = r < mask_prob_T
#             c += 1

#         if c > 1:
#             print(f"Generated mask {c} > 1 times.")

#         # x_THW = unfactorize_token_ids(x_THWC, config.num_factored_vocabs, config.factored_vocab_size)
#         x_THW[:, first_masked_frame:][mask] = mask_token_id

#         return {
#             "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
#             "labels": rearrange(labels, "b t h w -> b (t h w)"),
#         }

#     return collate_fn
