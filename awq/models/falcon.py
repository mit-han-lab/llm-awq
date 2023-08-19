from .base import BaseAWQForCausalLM
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconForCausalLM

class FalconAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "FalconDecoderLayer"

    @staticmethod
    def get_model_layers(model: FalconForCausalLM):
        return model.transformer.h
    
    @staticmethod
    def get_act_for_scaling(module: FalconDecoderLayer):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.dense_h_to_4h.out_features
        )
    
    @staticmethod
    def move_embed(model: FalconForCausalLM, device):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: FalconDecoderLayer, input_feat, module_kwargs):
        layers = []
        
        # Falcon 7B (older architecture)
        if module.config.num_attention_heads == 71:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.input_layernorm,
                layers=[module.mlp.dense_h_to_4h, module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        # Falcon 40B (newer architecture)
        else:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.ln_attn,
                layers=[module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

            # linear 2
            layers.append(dict(
                prev_op=module.ln_mlp,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat['mlp.dense_h_to_4h'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        return layers