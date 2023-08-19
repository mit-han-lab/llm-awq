from .base import BaseAWQForCausalLM
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomBlock

class BloomAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "BloomBlock"

    @staticmethod
    def get_model_layers(model: BloomForCausalLM):
        return model.transformer.h
    
    @staticmethod
    def get_act_for_scaling(module: BloomBlock):
        return dict(
            is_scalable=True,
            scale_name="mlp.gelu_impl",
            scale_layer=module.mlp.gelu_impl,
            scale_shape=module.mlp.dense_h_to_4h.out_features
        )
    
    @staticmethod
    def move_embed(model: BloomForCausalLM, device: str):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: BloomBlock, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attention.query_key_value],
            inp=input_feat['self_attention.query_key_value'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # linear 2
        layers.append(dict(
            prev_op=module.mlp.gelu_impl,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))

        return layers