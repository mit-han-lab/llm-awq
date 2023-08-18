from transformers import AutoConfig
from awq.models import *
from awq.models.base import BaseAWQForCausalLM

AWQ_CAUSAL_LM_MODEL_MAP = {
    "mpt": MptAWQForCausalLM,
    "llama": LlamaAWQForCausalLM,
    "opt": OptAWQForCausalLM
}

def check_and_get_model_type(model_dir, trust_remote_code=True):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type

class AutoAWQForCausalLM:
    default_q_config = {"zero_point": True, "q_group_size": 128}

    def __init__(self):
        raise EnvironmentError('You must instantiate AutoAWQForCausalLM with\n'
                               'AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.from_pretrained')
    
    @classmethod
    def from_pretrained(self, model_path, trust_remote_code=True) -> BaseAWQForCausalLM:
        model_type = check_and_get_model_type(model_path, trust_remote_code)

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path, model_type, trust_remote_code=trust_remote_code
        )

    @classmethod
    def from_quantized(self, quant_path, quant_filename, w_bit=4, q_config={}, 
                       device='balanced', trust_remote_code=True) -> BaseAWQForCausalLM:
        model_type = check_and_get_model_type(quant_path, trust_remote_code)
        q_config = q_config if q_config else self.default_q_config

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path, model_type, quant_filename, w_bit, q_config, device, trust_remote_code=trust_remote_code
        )