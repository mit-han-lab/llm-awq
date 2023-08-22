from .base import BaseAWQForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTDecoderLayer

class OptAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "OPTDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model: OPTForCausalLM):
        return model.model.decoder.layers
    
    @staticmethod
    def get_act_for_scaling(module: OPTDecoderLayer):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model: OPTForCausalLM, device: str):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: OPTDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.self_attn_layer_norm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        # attention out
        layers.append(dict(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.out_proj],
            inp=input_feat['self_attn.out_proj'],
        ))

        # linear 1
        layers.append(dict(
            prev_op=module.final_layer_norm,
            layers=[module.fc1],
            inp=input_feat['fc1'],
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.fc1,
            layers=[module.fc2],
            inp=input_feat['fc2'],
        ))

        return layers