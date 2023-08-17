import os
import torch
from awq.quantize.auto_clip import apply_clip
from awq.quantize.auto_scale import apply_scale
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

max_memory = [v.split(':') for v in (None or [])]
max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}

def get_awq_model(model):
    from awq.models import MptAWQForCausalLM

    if "mpt" in str(model.__class__).lower():
        return MptAWQForCausalLM()
    else:
        raise NotImplementedError(type(model))

def load_unquantized(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)

    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs)

    model.eval()

    return model, tokenizer

def load_search_result_into_memory(model, search_path):
    awq_results = torch.load(search_path, map_location="cpu")
            
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])

def run_search(model, dump_path):
    model, tokenizer = load_unquantized(model_path)
    awq_model = get_awq_model(model)
    awq_results = awq_model.quantize(model, tokenizer, w_bit=4, q_config=q_config, run_search=True, run_quant=False)

    dirpath = os.path.dirname(dump_path)
    os.makedirs(dirpath, exist_ok=True)
    torch.save(awq_results, dump_path)

def run_quant(model, search_path, dump_path):
    model, tokenizer = load_unquantized(model_path)
    load_search_result_into_memory(model, search_path)

    awq_model = get_awq_model(model)
    awq_model.quantize(model, w_bit=4, q_config=q_config, run_search=False, run_quant=True)

    dirpath = os.path.dirname(dump_path)
    os.makedirs(dirpath, exist_ok=True)
    torch.save(model.cpu().state_dict(), dump_path)

model_path = "./mpt-7b-8k-chat"
search_path = "./mpt-7b-8k-chat/mpt-7b-8k-chat-awq-search.pt"
quant_path = "./mpt-7b-8k-chat/mpt-7b-8k-chat-w4-g128.pt"
q_config = { "zero_point": True, "q_group_size": 128 }
