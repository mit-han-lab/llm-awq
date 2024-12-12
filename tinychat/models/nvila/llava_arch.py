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

import copy
import json
import logging
import os
import os.path as osp
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict, deque
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from transformers import AutoConfig, GenerationConfig, PreTrainedModel
from transformers.modeling_utils import ContextManagers, no_init_weights
from time import time
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.mm_utils import process_image, process_images
from llava.model.configuration_llava import LlavaConfig
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.utils import get_model_config

# from llava.train.sequence_parallel import get_pg_manager
from llava.utils import distributed as dist
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation
from .builder import build_tokenizer


class LlavaMetaModel(ABC):
    def init_vlm(self, config, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if (
            hasattr(self, "llm")
            or hasattr(self, "vision_tower")
            or hasattr(self, "mm_projector")
        ):
            # already initialized, skipped
            return

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn(
                "model_dtype not found in config, defaulting to torch.float16."
            )
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            self.llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config."
            )
        self.tokenizer = build_tokenizer(self.llm_cfg, config, *args, **kwargs)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.encoders = {}
        for name in ["image", "video"]:
            config = getattr(self.config, f"{name}_encoder")
            if isinstance(config, str):
                config = json.loads(config)
            self.encoders[name] = instantiate(config, parent=self)

        self.post_config()
        self.is_loaded = True

        assert (
            self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def load_from_config(cls, model_path_or_config, *args, **kwargs):
        pass

    ## FIXME we will use this function to load model in the future
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)

        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, LlavaConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(
                f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, LlavaConfig)}"
            )

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn(
                "model_dtype not found in config, defaulting to torch.float16."
            )
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config."
            )

        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained")
        init_context = [
            no_init_weights(_enable=True),
        ]
        # print("Before Init Context")
        # if hasattr(config, "deepspeed") and "mics" in config.deepspeed:
        #     print("Using MiCS_Init")
        #     import deepspeed
        #     init_context.append(deepspeed.zero.MiCS_Init(config_dict_or_path=config.deepspeed))
        with ContextManagers(init_context):
            vlm = cls(config, *args, **kwargs)
        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained finish")

        if (
            hasattr(vlm, "llm")
            or hasattr(vlm, "vision_tower")
            or hasattr(vlm, "mm_projector")
        ):
            if vlm.is_loaded:
                return vlm

        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(
            llm_cfg, config, *args, **kwargs
        )
        vlm.vision_tower = build_vision_tower(vision_tower_cfg, config)
        vlm.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.post_config()
        self.is_loaded = True

        # FIXME(ligeng, yunhao): llm should never be none here.
        assert (
            vlm.llm is not None
            or vlm.vision_tower is not None
            or vlm.mm_projector is not None
        ), "At least one of the components must be instantiated."
        return vlm

    ## FIXME we will use this function to save the model in the future
    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            # other wise fetch from deepspeed
            # state_dict = accelerator.get_state_dict(is_deepspeed_enabled)
            state_dict = self.state_dict()

        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict(
                {k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k}
            )
            self.llm.save_pretrained(
                os.path.join(output_dir, "llm"), state_dict=llm_state_dict
            )
            self.config.llm_cfg = self.llm.config

        if self.get_vision_tower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(
                output_dir, "vision_tower"
            )
            vision_tower_state_dict = OrderedDict(
                {
                    k.split("vision_tower.vision_tower.")[-1]: v
                    for k, v in state_dict.items()
                    if "vision_tower" in k
                }
            )
            self.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vision_tower.image_processor.save_pretrained(
                os.path.join(output_dir, "vision_tower")
            )
            self.config.vision_tower_cfg = self.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, "auto_map"):
                if "radio" not in self.get_vision_tower().__class__.__name__.lower():
                    delattr(self.config.vision_tower_cfg, "auto_map")

        if self.get_mm_projector():
            print(f"saving mm_projector to {osp.join(output_dir, 'mm_projector')}")
            self.mm_projector.config._name_or_path = osp.join(
                output_dir, "mm_projector"
            )
            mm_projector_state_dict = OrderedDict(
                {
                    k.split("mm_projector.")[-1]: v
                    for k, v in state_dict.items()
                    if "mm_projector" in k
                }
            )
            self.mm_projector.save_pretrained(
                os.path.join(output_dir, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            self.config.mm_projector_cfg = self.mm_projector.config
        ## update and save top-level config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def post_config(self):

        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config

    @staticmethod
    def merge_chessboard(x, num_split_h, num_split_w):
        """
        x: b * n * c or b * h * w * c
        out: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        """
        B = x.shape[0]
        if x.dim() == 3:
            N = x.shape[1]
            x = rearrange(x, "b (h w) c -> b c h w", h=int(N**0.5), w=int(N**0.5))

        assert B % (num_split_h * num_split_w) == 0
        b = B // (num_split_h * num_split_w)

        x_merge = torch.cat(
            [
                torch.cat(
                    [
                        x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b]
                        for j in range(num_split_w)
                    ],
                    dim=-1,
                )
                for i in range(num_split_h)
            ],
            dim=-2,
        )

        return x_merge

    @staticmethod
    def split_chessboard(x, num_split_h, num_split_w):
        """
        x: b * c * h * w
        out: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
        """
        B, C, H, W = x.shape
        assert H % num_split_h == 0 and W % num_split_w == 0
        h, w = H // num_split_h, W // num_split_w
        x_split = torch.cat(
            [
                x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]
                for i in range(num_split_h)
                for j in range(num_split_w)
            ],
            dim=0,
        )
        return x_split

    def merge_features_for_dynamic_s2(self, image_features, block_sizes):
        scales = self.get_vision_tower().scales
        resize_output_to_scale_idx = self.get_vision_tower().resize_output_to_scale_idx

        image_features_each_image = []
        new_block_sizes = []
        block_cnt = 0
        for block_size_each_image in block_sizes:
            if block_size_each_image is None:
                cur_features = image_features[block_cnt : block_cnt + 1]
                cur_features = rearrange(
                    cur_features,
                    "1 (h w) c -> 1 c h w",
                    h=int(cur_features.shape[1] ** 0.5),
                )
                cur_features = cur_features.repeat(1, len(scales), 1, 1)
                image_features_each_image.append(cur_features)
                new_block_sizes.append((1, 1))
                block_cnt += 1
            else:
                cur_features_each_scale = []
                for scale in scales[:-1]:
                    num_blocks_this_scale = (scale // scales[0]) ** 2
                    cur_features_each_scale.append(
                        self.merge_chessboard(
                            image_features[
                                block_cnt : block_cnt + num_blocks_this_scale
                            ],
                            num_split_h=scale // scales[0],
                            num_split_w=scale // scales[0],
                        )
                    )  # 1 * C * H * W
                    block_cnt += num_blocks_this_scale
                num_blocks_last_scale = (
                    block_size_each_image[0] * block_size_each_image[1]
                )
                cur_features_each_scale.append(
                    self.merge_chessboard(
                        image_features[block_cnt : block_cnt + num_blocks_last_scale],
                        num_split_h=block_size_each_image[0],
                        num_split_w=block_size_each_image[1],
                    )
                )  # 1 * C * H * W
                block_cnt += num_blocks_last_scale

                # resize and concat features from different scales
                output_size = cur_features_each_scale[resize_output_to_scale_idx].shape[
                    -2:
                ]
                cur_features = torch.cat(
                    [
                        F.interpolate(
                            cur_features_each_scale[i].to(torch.float32),
                            size=output_size,
                            mode="area",
                        ).to(cur_features_each_scale[i].dtype)
                        for i in range(len(cur_features_each_scale))
                    ],
                    dim=1,
                )
                # cur_features = rearrange(cur_features, "1 c h w -> (h w) c")

                image_features_each_image.append(cur_features)

                if (
                    resize_output_to_scale_idx == len(scales) - 1
                    or resize_output_to_scale_idx == -1
                ):
                    new_block_sizes.append(block_size_each_image)
                else:
                    new_block_sizes.append(
                        (
                            scales[resize_output_to_scale_idx] // scales[0],
                            scales[resize_output_to_scale_idx] // scales[0],
                        )
                    )

        assert block_cnt == len(image_features)

        return image_features_each_image, new_block_sizes

    def encode_images(
        self, images, block_sizes: Optional[Optional[Tuple[int, ...]]] = None
    ):
        if block_sizes is None:
            block_sizes = [None] * len(images)
        if getattr(self.config, "dynamic_s2", False):
            image_features = self.get_vision_tower()(images)
            image_features, new_block_sizes = self.merge_features_for_dynamic_s2(
                image_features, block_sizes
            )

            image_features = [
                self.split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes)
            ]  # list of B * C * H * W tensors
            image_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in image_features], dim=0
            )  # B * N * C
            image_features = self.get_mm_projector()(image_features)
            image_features = list(
                image_features.split(
                    [block_size[0] * block_size[1] for block_size in new_block_sizes],
                    dim=0,
                )
            )
            image_features = [
                self.merge_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes)
            ]  # list of 1 * C * H * W tensors
            image_features = [
                rearrange(x, "1 c h w -> (h w) c") for x in image_features
            ]  # list of N * C tensors
            image_features = torch.stack(image_features, dim=0)
        else:
            image_features = self.get_vision_tower()(images)
            image_features = self.get_mm_projector()(image_features)
        return image_features

    ## @yunhao: is there a better way to handle function call and attributes for llm?
    ## support beam search
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)


class LlavaMetaForCausalLM(ABC):
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = (
            labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        )
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        media_embeds = self.__embed_media_tokens(media, media_config)

        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0
        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    end = pos + 1
                    name = media_tokens[input_ids[k][pos].item()]
                    input = media_embeds[name].popleft()
                    label = torch.full(
                        [input.shape[0]],
                        IGNORE_INDEX,
                        device=labels[k].device,
                        dtype=labels[k].dtype,
                    )
                else:
                    end = pos
                    while (
                        end < len(labels[k])
                        and input_ids[k][end].item() not in media_tokens
                    ):
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]
                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed
        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed!")

        # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return self.__batchify_sequence(inputs, labels)

    def __embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)
        for name in media:
            embeds[name] = deque(self.encoders[name](media[name], media_config[name]))
        return embeds

    def __truncate_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if any(len(input) > self.tokenizer.model_max_length for input in inputs):
            warnings.warn(
                f"Truncating sequences to `model_max_length` ({self.tokenizer.model_max_length})."
            )
            inputs = [input[: self.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.tokenizer.model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones(
            (batch_size, max_length), dtype=torch.bool, device=device
        )

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros(
                (size_pk, hidden_size), dtype=inputs[k].dtype, device=device
            )
            labels_pk = torch.full(
                (size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device
            )
            if self.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        quant_llm: Optional[bool] = True,
        **generation_kwargs,
    ):
        inputs_embeds, _, attention_mask = self._embed(
            input_ids, media, media_config, None, attention_mask
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            quant_llm=quant_llm,
            **generation_kwargs,
        )

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        quant_llm: Optional[bool] = True,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Extract media from the conversation

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in [
                    "dynamic",
                    "dynamic_s2",
                ]:
                    self.config.image_processor = self.vision_tower.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(
                            media["image"][0],
                            self.config,
                            None,
                            enable_dynamic_res=True,
                        ).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN,
                            f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0],
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(
                                map(int, self.config.s2_scales.split(","))
                            )
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(
                        media["image"], self.vision_tower.image_processor, self.config
                    ).half()
                media[name] = [image for image in images]
            elif name == "video":
                media[name] = [
                    process_images(
                        images, self.vision_tower.image_processor, self.config
                    ).half()
                    for images in media[name]
                ]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        # Tokenize the conversation
        input_ids = (
            tokenize_conversation(
                conversation, self.tokenizer, add_generation_prompt=True
            )
            .cuda()
            .unsqueeze(0)
        )

        # Set up the generation config
        generation_config = generation_config or self.default_generation_config
        # Generate the response
        try:
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
                quant_llm=quant_llm,
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            # FIXME(zhijianl): This is a temporary workaround for the sampling issue
            logging.warning(
                "Generation failed with sampling, retrying with greedy decoding."
            )
            generation_config.do_sample = False
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
            )

        # Decode the response
        response = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
        return response

    @torch.inference_mode()
    def benchmark(self, prompt: Union[str, List], quant_llm) -> None:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Extract media from the conversation

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        image_num = 0
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in [
                    "dynamic",
                    "dynamic_s2",
                ]:
                    self.config.image_processor = self.vision_tower.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(
                            media["image"][0],
                            self.config,
                            None,
                            enable_dynamic_res=True,
                        ).half()
                        if len(images.shape) == 3:
                            images = images.reshape(1, *images.shape)
                        image_num += images.shape[0]
                        size = images.shape[1:]
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN,
                            f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0],
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(
                                map(int, self.config.s2_scales.split(","))
                            )
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        if len(images.shape) == 3:
                            images = images.reshape(1, *images.shape)
                        image_num += images.shape[0]
                        size = images.shape[1:]
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(
                        media["image"], self.vision_tower.image_processor, self.config
                    ).half()
                    image_num += images.shape[0]
                    size = images.shape[1:]
                media[name] = [image for image in images]
            elif name == "video":
                media[name] = [
                    process_images(
                        images, self.vision_tower.image_processor, self.config
                    ).half()
                    for images in media[name]
                ]
                for images in media[name]:
                    image_num += images.shape[0]
                    size = images.shape[1:]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        # Tokenize the conversation
        input_ids = (
            tokenize_conversation(
                conversation, self.tokenizer, add_generation_prompt=True
            )
            .cuda()
            .unsqueeze(0)
        )

        # Set up the generation config
        for i in range(10):
            torch.cuda.synchronize()
            t_st = time()
            inputs_embeds, _, attention_mask = self._embed(
                input_ids, media, media_config, None, None
            )
            torch.cuda.synchronize()
            t_ed = time()
            torch.cuda.empty_cache()
        print(
            "Time of vision tower and others is {:.5f} s for {} images ({} x {} x {})".format(
                t_ed - t_st, image_num, size[0], size[1], size[2]
            )
        )
        output = self.llm.benchmark(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            quant_llm=quant_llm,
        )
        # response = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        return

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = (
                self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            )
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.stop_token_ids
        return generation_config

    # Prepare media

    # Process media
    @torch.inference_mode()
    def prepare_media(self, conversation):
        media = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in [
                    "dynamic",
                    "dynamic_s2",
                ]:
                    self.config.image_processor = self.vision_tower.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(
                            media["image"][0],
                            self.config,
                            None,
                            enable_dynamic_res=True,
                        ).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN,
                            f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0],
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(
                                map(int, self.config.s2_scales.split(","))
                            )
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(
                        media["image"], self.vision_tower.image_processor, self.config
                    ).half()
                media[name] = [image for image in images]
            elif name == "video":
                media[name] = [
                    process_images(
                        images, self.vision_tower.image_processor, self.config
                    ).half()
                    for images in media[name]
                ]
            else:
                raise ValueError(f"Unsupported media type: {name}")
        return media, media_config

    @torch.inference_mode()
    def stream_gen(
        self,
        input_ids,
        media,
        media_cfg,
        start_pos,
        chunk_prefilling,
        quant_llm,
        attention_mask=None,
    ) -> str:
        if media is None:
            inputs_embeds = self.llm.model.embed_tokens(input_ids)
        else:
            image_num = torch.sum(input_ids == 151649)
            if image_num == 1 and self.config.image_aspect_ratio == "dynamic":
                patch_num = len(media["image"])
                new_input_ids = []
                for i, id in enumerate(input_ids[0]):
                    if id == 151649:
                        new_input_ids.extend(input_ids[0, 0:i])
                        new_input_ids.extend([198, 151649, 198] * patch_num)
                        new_input_ids.extend(input_ids[0, i + 1 :])
                        break
                input_ids = torch.tensor(
                    [new_input_ids], dtype=torch.int, device="cuda"
                )
            inputs_embeds, _, _ = self._embed(
                input_ids, media, media_cfg, None, attention_mask=None
            )
        length = inputs_embeds.shape[1]
        if quant_llm:
            out = self.llm(None, start_pos, inputs_embeds, chunk_prefilling)
        else:
            out = self.llm.forwardfp16(None, start_pos, inputs_embeds, chunk_prefilling)
        return out, length
