import os
import time
import torch
import argparse
from lm_eval import evaluator
from transformers import AutoTokenizer
from awq.models.auto import AutoAWQForCausalLM
from awq.quantize.auto_clip import apply_clip
from awq.quantize.auto_scale import apply_scale
from awq.utils.lm_eval_adaptor import LMEvalAdaptor


def load_search_result_into_memory(model, search_path):
    awq_results = torch.load(search_path, map_location="cpu")
            
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])

def run_search(model_path, dump_path, quant_config):
    """
    Step 1/2: Search the pile for an optimal scaling factor.
    """
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, run_search=True, run_quant=False)

    # Save search results
    model.save_quantized(dump_path)

    # Save tokenizer
    tokenizer.save_pretrained(dump_path)

def run_quant(model_path, search_path, dump_path, quant_config):
    """
    Step 2/2: Use the search results to quantize model weights
    """
    # Load model and search results
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    load_search_result_into_memory(model.model, search_path)

    # Run actual weight quantization
    model.quantize(quant_config=quant_config, run_search=False, run_quant=True)

    # Save quantized model
    model.save_quantized(dump_path)

def run_eval(model_path, quant_file, device, tasks, task_batch_size, task_n_shot, task_use_pretrained):
    """
    Post quantization: Evaluate perplexity on wikitext with EleutherAI Evaluation Harness
    """
    # Load model
    if task_use_pretrained:
        model = AutoAWQForCausalLM.from_pretrained(model_path)
    else:
        model = AutoAWQForCausalLM.from_quantized(model_path, quant_file)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load adapter
    lm_eval_model = LMEvalAdaptor(model_path, model, tokenizer, device, batch_size=task_batch_size)

    # Evaluate perplexity of quantized model
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks.split(','),
        batch_size=task_batch_size,
        no_cache=True,
        num_fewshot=task_n_shot,
    )

    print(evaluator.make_table(results))

@torch.inference_mode()
def run_speed(model_path, quant_file, device, n_generate=128, max_new_tokens=256):
    def _timer(func):
        start = time.time()
        out = func()
        return out, time.time() - start
    
    def _generate(model, model_out, n_generate):
        past_key_values = model_out.past_key_values

        for i in range(n_generate):
            logits = model_out.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            token = torch.as_tensor([token], device=device).unsqueeze(0)

            model_out = model(token, use_cache=True, past_key_values=past_key_values)
    
    def _warmup(device:str):
        warm_up = torch.randn((4096,4096)).to(device)
        torch.mm(warm_up,warm_up)
    
    # Load model
    model, load_time = _timer(lambda: AutoAWQForCausalLM.from_quantized(model_path, quant_file, fuse_layers=True))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _warmup(device)

    # Generate random inputs
    n_context = max_new_tokens - n_generate
    ids = torch.randint(0, tokenizer.vocab_size, (1, n_context)).cuda()

    # Context stage
    model_out, context_time = _timer(lambda: model(ids, use_cache=True))

    # Generation stage
    _, generation_time = _timer(lambda: _generate(model, model_out, n_generate))
    
    # Prints
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    context_tokens_per_second = n_context / context_time
    context_ms_per_token = (context_time*1000) / n_context
    inference_tokens_per_second = n_generate / generation_time
    inference_ms_per_token = (generation_time*1000) / n_generate

    print(f"[======] Model summary: {model_path} [======]")
    print(f"[*] Load time: {load_time:.2f} seconds")
    print(f"[*] Context speed: {context_tokens_per_second:.2f} tokens/second ({context_ms_per_token:.2f} ms/token)")
    print(f"[*] Generation speed: {inference_tokens_per_second:.2f} tokens/second ({inference_ms_per_token:.2f} ms/token)")
    print(f"[*] VRAM: {memory_used:.2f} MB")

if __name__ == '__main__':
    """
    - Run AWQ search and save result:
    python -m awq.entry --entry_type search --model_path lmsys/vicuna-7b-v1.5 --search_path vicuna-7b-v1.5-awq

    - Run AWQ to save the real quantized weights at the quant_path:
    python -m awq.entry --entry_type quant --model_path lmsys/vicuna-7b-v1.5 --search_path vicuna-7b-v1.5-awq/awq_model_search_result.pt --quant_path vicuna-7b-v1.5-awq

    - Run perplexity of quantized model:
    python -m awq.entry --entry_type eval --model_path vicuna-7b-v1.5-awq --quant_file awq_model_w4_g128.pt

    - Run perplexity unquantized FP16 model:
    python -m awq.entry --entry_type eval --model_path lmsys/vicuna-7b-v1.5 --task_use_pretrained

    - Run a speedtest to benchmark the quantized model:
    python -m awq.entry --entry_type speed --model_path vicuna-7b-v1.5-awq --quant_file awq_model_w4_g128.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry_type', type=str, help='The type of task to run (search|quant|eval|speed)')
    parser.add_argument('--model_path', type=str, help='Path to hf model')
    parser.add_argument('--search_path', type=str, help='Path to save/load AWQ search results')
    parser.add_argument('--quant_path', type=str, help='Path to save AWQ model to directory')
    parser.add_argument('--quant_file', type=str, help='Path to quantized AWQ model file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to load model to')
    parser.add_argument('--w_bit', type=int, default=4)
    parser.add_argument('--q_group_size', type=int, default=128)
    parser.add_argument('--tasks', type=str, default='wikitext', help='Tasks to evaluate. '
                        'Separate tasks by comma for multiple tasks.'
                        'https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md')
    parser.add_argument("--task_use_pretrained", default=False, action=argparse.BooleanOptionalAction,
                        help="Pass '--task_use_pretrained' to use a pretrained model running FP16")
    parser.add_argument('--task_batch_size', type=int, default=1)
    parser.add_argument('--task_n_shot', type=int, default=0)
    parser.add_argument('--n_generate', type=int, default=128)
    parser.add_argument('--n_context', type=int, default=256)
    args = parser.parse_args()

    quant_config = { "zero_point": True, "q_group_size": args.q_group_size, "w_bit": args.w_bit }
    
    if args.entry_type == 'search':
        run_search(args.model_path, args.search_path, quant_config)
    elif args.entry_type == 'quant':
        run_quant(args.model_path, args.search_path, args.quant_path, quant_config)
    elif args.entry_type == 'eval':
        run_eval(args.model_path, args.quant_file, args.device, 
                       args.tasks, args.task_batch_size, args.task_n_shot, args.task_use_pretrained)
    elif args.entry_type == 'speed':
        run_speed(args.model_path, args.quant_file, args.device, args.n_generate, args.n_context)
    else:
        raise Exception('--entry_type must be one of (search|quant|eval|speed)')