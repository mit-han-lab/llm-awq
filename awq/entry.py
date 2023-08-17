import os
import torch
import argparse
from lm_eval import evaluator
from awq.quantize.auto_clip import apply_clip
from awq.quantize.auto_scale import apply_scale
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

def load_quantized(model_path, quant_path, w_bit, q_config, device):
    from awq.models.auto import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_quantized(model_path, quant_path, w_bit, q_config, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer

def load_search_result_into_memory(model, search_path):
    awq_results = torch.load(search_path, map_location="cpu")
            
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])

def run_search(model_path, dump_path, w_bit, q_config):
    model, tokenizer = load_unquantized(model_path)
    awq_model = get_awq_model(model)
    awq_results = awq_model.quantize(model, tokenizer, w_bit=w_bit, q_config=q_config, run_search=True, run_quant=False)

    dirpath = os.path.dirname(dump_path)
    os.makedirs(dirpath, exist_ok=True)
    torch.save(awq_results, dump_path)

def run_quant(model_path, search_path, dump_path, w_bit, q_config, device):
    model, tokenizer = load_unquantized(model_path, device)
    load_search_result_into_memory(model, search_path)

    awq_model = get_awq_model(model)
    awq_model.quantize(model, w_bit=w_bit, q_config=q_config, run_search=False, run_quant=True)

    dirpath = os.path.dirname(dump_path)
    os.makedirs(dirpath, exist_ok=True)
    torch.save(model.cpu().state_dict(), dump_path)

def run_perplexity(model_path, quant_path, w_bit, q_config, device):
    model, tokenizer = load_quantized(model_path, quant_path, w_bit, q_config, device)

    lm_eval_model = LMEvalAdaptor(model_path, model, tokenizer, device, batch_size=1)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=['wikitext'],
        batch_size=1,
        no_cache=True,
        num_fewshot=0,
    )

    print(evaluator.make_table(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry_type', type=str, help='The type of task to run (search|quant|perplexity)')
    parser.add_argument('--model_path', type=str, help='Path to hf model')
    parser.add_argument('--search_path', type=str, help='Path to save/load AWQ search results')
    parser.add_argument('--quant_path', type=str, help='Path to save/load AWQ quant model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to load model to')
    parser.add_argument('--w_bit', type=int, default=4)
    parser.add_argument('--q_group_size', type=int, default=128)
    args = parser.parse_args()

    args.model_path = "./mpt-7b-8k-chat"
    args.search_path = "./mpt-7b-8k-chat/mpt-7b-8k-chat-awq-search.pt"
    args.quant_path = "./mpt-7b-8k-chat/mpt-7b-8k-chat-w4-g128.pt"
    q_config = { "zero_point": True, "q_group_size": args.q_group_size }
    
    if args.entry_type == 'search':
        run_search(args.model_path, args.search_path, args.w_bit, q_config)
    elif args.entry_type == 'quant':
        run_quant(args.model_path, args.search_path, args.quant_path, args.w_bit, q_config)
    elif args.entry_type == 'perplexity':
        run_perplexity(args.model_path, args.quant_path, args.w_bit, q_config, args.device)
    else:
        raise Exception('--entry_type must be one of (search|quant|perplexity)')