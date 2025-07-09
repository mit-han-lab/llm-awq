import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from time import time

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import (AutoConfig, 
                          AutoModel, 
                          AutoTokenizer,
                          GenerationConfig,
                          PretrainedConfig, 
                          PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import modeling_utils

from .internvl.configuration_internvl import InternVisionConfig, InternVLChatConfig
from .internvl.internvit import InternVisionModel
from .internvl.conversation import get_conv_template
from .internvl.media import load_image, load_video

from llava.media import Image, Video

from .qwen2 import Qwen2ForCausalLM
from .llama import LlamaForCausalLM

try:
    import flash_attn
    has_flash_attn = True
except ImportError:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False

def skip(*args, **kwargs):
    pass

torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.kaiming_normal_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

modeling_utils._init_weights = False


logger = logging.get_logger(__name__)


class InternVL3(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']
        
    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path, trust_remote_code=True, use_fast=False)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
    
    def freezed_module_patch(self):
        self.vision_model.eval()
        self.language_model.eval()
        self.mlp1.eval()
    
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    @torch.inference_mode()
    def prepare_media(self, conversation):
        prompt = conversation[0]["value"]
        media = {"image": [], "video": []}
        for item in prompt:
            if isinstance(item, Image):
                media["image"].append(load_image(item.path))
            if isinstance(item, Video):
                pixel_values, num_patches_list = load_video(item.path)
                media["video"].extend(pixel_values)
        
        return media, num_patches_list if media["video"] else None
        
    @torch.inference_mode()
    def extract_features(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
        
    @torch.inference_mode()
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ):
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        
        if media["image"]:
            pixel_values = torch.cat(media["image"], dim=0).half().cuda()
        elif media["video"]:
            pixel_values = torch.cat(media["video"], dim=0).half().cuda()
            
        vit_embeds = self.extract_features(pixel_values)
        
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        
        input_embeds = input_embeds.reshape(B, N, C)
        
        return input_embeds, None, attention_mask
    
    @torch.inference_mode()
    def benchmark(self, prompt: Union[str, List], quant_llm) -> None:
        media = {"image": [], "video": []}
        question = ""
        for item in prompt:
            if isinstance(item, str):
                question += item
            if isinstance(item, Image):
                media["image"].append(load_image(item.path))
            if isinstance(item, Video):
                pixel_values, num_patches_list = load_video(item.path)
                media["video"].extend(pixel_values)
        
        if media["image"]:
            num_patches_list = [image.size(0) for image in media["image"]]
        
        if media["image"] and '<image>' not in question:
            question = '<image>\n' + question
        
        if media["video"] and '<image>' not in question:
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + question
        
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            
        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        
        for i in range(10):
            torch.cuda.synchronize()
            t_st = time()
            inputs_embeds, _, attention_mask = self._embed(
                input_ids=input_ids, 
                media=media,
                media_config=None,
                labels=None,
                attention_mask=attention_mask
            )
            torch.cuda.synchronize()
            t_ed = time()
            torch.cuda.empty_cache()
            
        if media["image"]:
            print(
                "Time of vision tower and others is {:.5f} s for {} images ({} x {} x {})".format(
                    t_ed - t_st, sum(num_patches_list), media["image"][0].shape[1], media["image"][0].shape[2], media["image"][0].shape[3]
                )
            )
        elif media["video"]:
            print(
                "Time of vision tower and others is {:.5f} s for {} video frames ({} x {} x {})".format(
                    t_ed - t_st, sum(num_patches_list),  media["video"][0].shape[1], media["video"][0].shape[2], media["video"][0].shape[3]
                )
            )
        output = self.language_model.benchmark(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            quant_llm=quant_llm
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        
        return response

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
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids ).clone()
        else:
            inputs_embeds, _, _ = self._embed(input_ids, media, None, None, attention_mask)
    
        length = inputs_embeds.shape[1]
        if quant_llm:
            out = self.language_model(None, start_pos, inputs_embeds, chunk_prefilling)
        else:
            out = self.language_model.forwardfp16(None, start_pos, inputs_embeds, chunk_prefilling)
        return out, length

    @torch.inference_mode()
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
