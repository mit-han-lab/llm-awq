from transformers import AutoConfig
from awq.models import *
from awq.models.base import BaseAWQForCausalLM

AWQ_CAUSAL_LM_MODEL_MAP = {
    "mpt": MptAWQForCausalLM,
    "llama": LlamaAWQForCausalLM,
    "opt": OptAWQForCausalLM,
    "RefinedWeb": FalconAWQForCausalLM,
    "RefinedWebModel": FalconAWQForCausalLM,
    "bloom": BloomAWQForCausalLM
}

def check_and_get_model_type(model_dir, trust_remote_code=True):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type

class AutoAWQForCausalLM:
    def __init__(self):
        raise EnvironmentError('You must instantiate AutoAWQForCausalLM with\n'
                               'AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.from_pretrained')
    
    @classmethod
    def from_pretrained(self, model_path, trust_remote_code=True, safetensors=False) -> BaseAWQForCausalLM:
        model_type = check_and_get_model_type(model_path, trust_remote_code)

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path, model_type, trust_remote_code=trust_remote_code, safetensors=safetensors
        )

    @classmethod
    def from_quantized(self, quant_path, quant_filename, max_new_tokens=None,
                       device='balanced', trust_remote_code=True) -> BaseAWQForCausalLM:
        model_type = check_and_get_model_type(quant_path, trust_remote_code)

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path, model_type, quant_filename, max_new_tokens, device, trust_remote_code=trust_remote_code
        )