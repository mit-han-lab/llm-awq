from transformers import AutoConfig
from awq.models import MptAWQForCausalLM

AWQ_CAUSAL_LM_MODEL_MAP = {
    "mpt": MptAWQForCausalLM,
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
    def from_pretrained():
        pass

    @classmethod
    def from_quantized(self, model_path, quant_path, w_bit, q_config, device, trust_remote_code=True):
        model_type = check_and_get_model_type(model_path, trust_remote_code)

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type]().from_quantized(
            model_path, quant_path, w_bit, q_config, device
        )