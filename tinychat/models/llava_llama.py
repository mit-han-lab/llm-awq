#    Modified from https://github.com/haotian-liu/LLaVA
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import CLIPVisionModel

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_base.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llama import LlamaForCausalLM, Transformer


class LlavaLlamaModel(LlavaMetaModel, Transformer):
    def __init__(self, config):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    def __init__(self, config, dev="cuda"):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        self.device = dev

    def get_model(self):
        return self.model

    def default_inputs_embeds_for_multimodal(self, input_ids, inputs_embeds, images):
        if inputs_embeds is None:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        vision_tower = self.get_vision_tower().vision_tower
        from contextlib import nullcontext

        if (
            vision_tower is not None
            and (input_ids.shape[1] != 1 or self.training)
            and images is not None
        ):
            from tinychat.utils.constants import LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX

            with nullcontext() if getattr(
                self.config, "tune_vision_encoder", False
            ) else torch.no_grad():
                if type(images) is list:
                    images = [
                        image.unsqueeze(0) if len(image.shape) == 3 else image
                        for image in images
                    ]
                    images = torch.cat(images, dim=0)
                dtype = next(vision_tower.parameters()).dtype
                if "visiontransformer" in vision_tower.__class__.__name__.lower():
                    image_features = vision_tower(images.to(dtype))
                else:
                    image_forward_outs = vision_tower(
                        images.to(dtype), output_hidden_states=True
                    )
                    select_hidden_state_layer = getattr(
                        self.config, "mm_vision_select_layer", -1
                    )
                    if abs(select_hidden_state_layer) > 100:  # TOOD: find a better impl
                        # -212 -> 12,
                        idx1, idx2 = abs(select_hidden_state_layer) % 100, -(
                            abs(select_hidden_state_layer) // 100
                        )
                        # print("selecting multiple indices", idx1, idx2)
                        image_features = torch.cat(
                            (
                                image_forward_outs.hidden_states[idx1],
                                image_forward_outs.hidden_states[idx2],
                            ),
                            dim=-1,
                        )
                    else:
                        image_features = image_forward_outs.hidden_states[
                            select_hidden_state_layer
                        ]
                if isinstance(vision_tower, CLIPVisionModel):  # clip case, not for sam
                    image_features = image_features[:, 1:].to(images.dtype)  # (B, N, D)

            image_features = self.model.mm_projector(image_features)

            if hasattr(self.config, "neftune_alpha") and self.config.neftune_alpha > 0:
                # print("using neftune tuning with alpha", self.config.neftune_alpha)
                dims = torch.tensor(image_features.shape[-2] * image_features.shape[-1])
                mag_norm = self.config.neftune_alpha / torch.sqrt(dims)
                image_features = image_features + torch.zeros_like(
                    image_features
                ).uniform_(-mag_norm, mag_norm)

            if self.config.mm_projector_type == "dsresampler":
                dummy_feat_shape = (1, 1024, 1664)
            elif self.config.mm_projector_type == "linear2":
                dummy_feat_shape = (1, 256, self.config.mm_hidden_size * 2)
            else:
                dummy_feat_shape = (1, 256, self.config.mm_hidden_size)

            dummy_image_features = torch.zeros(
                *dummy_feat_shape,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy_image_features = self.model.mm_projector(dummy_image_features)[
                0
            ]  # (1, N, D)

            new_input_embeds = []
            cur_image_idx = 0

            image_token_idx = []

            num_patches = -1
            for i_sample, (cur_input_ids, cur_input_embeds) in enumerate(
                zip(input_ids, inputs_embeds)
            ):
                if (cur_input_ids == LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_image_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    # cur_image_idx += 1
                    continue
                # TODO: Need to fix if vision_tower.config.use_im_start_end == True
                num_total_patches = (
                    (cur_input_ids == LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX).sum().item()
                )
                masked_indices = torch.where(
                    cur_input_ids == LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX
                )[0]

                while num_total_patches:
                    if cur_image_idx >= image_features.shape[0]:  # SHOULD NOT HAPPEN!!!
                        if self.training:
                            print("%" * 20, "INDEXING ERROR!")
                            break
                        else:
                            raise ValueError("INDEXING ERROR!")
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    mask_index_start = masked_indices[0]
                    masked_indices = masked_indices[num_patches:]

                    image_token_idx.append(
                        (
                            i_sample,
                            mask_index_start.item(),
                            (mask_index_start + num_patches).item(),
                        )
                    )

                    orig_embeds_params = None
                    if orig_embeds_params is not None:
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_image_features,
                                cur_input_embeds[
                                    mask_index_start + num_patches :
                                ].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_image_features,
                                cur_input_embeds[mask_index_start + num_patches :],
                            ),
                            dim=0,
                        )
                    num_total_patches -= num_patches
                    assert num_total_patches >= 0, (num_total_patches, num_patches)
                    cur_image_idx += 1

                new_input_embeds.append(cur_input_embeds)
                if self.training:
                    if not masked_indices.numel() == 0:
                        print("%" * 20, "ERROR! masked_indices not empty...")
                else:
                    assert masked_indices.numel() == 0

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        start_pos: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        special_token: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            if special_token:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )
            else:
                inputs_embeds = self.default_inputs_embeds_for_multimodal(
                    input_ids, inputs_embeds, images
                )
                input_ids = None

        if start_pos == None:
            out = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            out = super().forward(
                tokens=input_ids,
                start_pos=start_pos,
                inputs_embeds=inputs_embeds,
            )
        return out

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs
