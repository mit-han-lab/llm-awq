import os
import warnings
import shutil
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.utils import get_model_config
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llama import LlamaForCausalLM, Transformer


class VilaLlamaForCausalLM(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.init_vlm(config)
    
    def init_vlm(self, config = None, *args, **kwargs):
        if hasattr(self, "llm") or hasattr(self, "vision_tower")  or hasattr(self, "mm_projector"):
            # already initialized, skipped
            return 
        
        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype
        
        # print("init_vlm(): config", config); input("DEBUG init_vlm")
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")
        # print("init_vlm():", cfgs); input("DEBUG init_vlm")
        llm_cfg = AutoConfig.from_pretrained(llm_cfg)
        
        # self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.llm = LlamaForCausalLM(llm_cfg)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."
    
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
        self.freezed_module_patch()
        if inputs_embeds is None:
            (
                _,
                _,
                _,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        if inputs_embeds is not None:
            outputs = self.llm.forward(
                tokens=None,
                start_pos=start_pos,
                inputs_embeds=inputs_embeds,
            )
        else:
            outputs = self.llm.forward(
                tokens=input_ids,
                start_pos=start_pos,
                inputs_embeds=None,
            )
        return outputs