import torch
import argparse
import numpy as np
from awq.models import *
from awq.models.auto import AutoAWQForCausalLM
from attributedict.collections import AttributeDict
from tinychat.utils.prompt_templates import get_prompter, get_stop_token_ids
from tinychat.stream_generators import StreamGenerator, FalconStreamGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# opt_params in TinyLLMEngine
gen_params = AttributeDict([
                    ("seed", -1),               # RNG seed
                    ("n_threads", 1),           # TODO: fix this
                    ("n_predict", 512),         # new tokens to predict
                    ("n_parts", -1),            # amount of model parts (-1: determine from model dimensions)
                    ("n_ctx", 512),             # context size
                    ("n_batch", 512),           # batch size for prompt processing (must be >=32 to use BLAS)
                    ("n_keep", 0),              # number of tokens to keep from initial prompt
                    ("n_vocab", 50272),         # vocabulary size

                    # sampling parameters
                    ("logit_bias", dict()),     # logit bias for specific tokens: <int, float>
                    ("top_k", 40),              # <= 0 to use vocab size
                    ("top_p", 0.95),            # 1.0 = disabled
                    ("tfs_z", 1.00),            # 1.0 = disabled
                    ("typical_p", 1.00),        # 1.0 = disabled
                    ("temp", 0.70),             # 1.0 = disabled
                    ("repeat_penalty", 1.10),   # 1.0 = disabled
                    ("repeat_last_n", 64),      # last n tokens to penalize (0 = disable penalty, -1 = context size)
                    ("frequency_penalty", 0.00),# 0.0 = disabled
                    ("presence_penalty", 0.00), # 0.0 = disabled
                    ("mirostat", 0),            # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
                    ("mirostat_tau", 5.00),     # target entropy
                    ("mirostat_eta", 0.10),     # learning rate
                ])

def stream_output(output_stream):
    print(f"ASSISTANT: ", end="", flush=True)
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    print(" ".join(output_text[pre:]), flush=True)
    if "timing" in outputs and outputs["timing"] is not None:
        timing = outputs["timing"]
        context_tokens = timing["context_tokens"]
        context_time = timing["context_time"]
        total_tokens = timing["total_tokens"]
        generation_time_list = timing["generation_time_list"]
        generation_tokens = len(generation_time_list)
        average_speed = (context_time + np.sum(generation_time_list)) / (context_tokens + generation_tokens)
        print("=" * 50)
        print("Speed of Inference")
        print("-" * 50)
        # print(f"Context Stage    : {context_time/context_tokens * 1000:.2f} ms/token")
        print(f"Generation Stage : {np.average(generation_time_list) * 1000:.2f} ms/token")
        # print(f"Average Speed    : {average_speed * 1000:.2f} ms/token")
        print("=" * 50)
        # print("token num:", total_tokens)
        # print("Model total Time = ", (context_time + np.sum(generation_time_list))*1000, "ms" )
    return " ".join(output_text)

def device_warmup(device:str):
    warm_up = torch.randn((4096,4096)).to(device)
    torch.mm(warm_up,warm_up)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/llm/checkpoints/vicuna-hf/vicuna-7b', help='path to the model')
    parser.add_argument('--quant_file', type=str, default='awq_model_w4_g128.pt', help='path to the model file')
    parser.add_argument('--precision' , type=str, default='W4A16', help='compute precision')
    parser.add_argument('--device'    , type=str, default='cuda')

    args = parser.parse_args()
    assert args.precision in ["W4A16", "W16A16"], "We only support W4A16/W16A16 now"

    gen_params.n_predict = 512
    gen_params.n_vocab = 32000

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if "mpt" in config.__class__.__name__.lower():
        # config.init_device="meta"
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    if args.precision == "W4A16":
        model = AutoAWQForCausalLM.from_quantized(args.model_path, args.quant_file)
        assert model.model_type.lower() in ["llama", "refinedweb", "refinedwebmodel", "mpt"], "We only support llama & falcon & mpt now"
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True).to(args.device)

    # device warm up
    device_warmup(args.device)

    if isinstance(model, FalconAWQForCausalLM):
        stream_generator = FalconStreamGenerator
    else:
        stream_generator = StreamGenerator

    # Optimize AWQ quantized model
    if args.precision == "W4A16" and isinstance(model, LlamaAWQForCausalLM):
        from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp
        make_quant_attn(model.model, args.device)
        make_quant_norm(model.model)
        make_fused_mlp(model.model)

    model_prompter = get_prompter(model, args.model_path)
    stop_token_ids = get_stop_token_ids(model, args.model_path) 
    count = 0
    while True:
        # Get input from the user
        input_prompt = input("USER: ")
        if input_prompt == "":
            print("EXIT...")
            break
        model_prompter.insert_prompt(input_prompt)
        output_stream = stream_generator(model, tokenizer, model_prompter.model_input, gen_params, device=args.device, stop_token_ids = stop_token_ids)
        outputs = stream_output(output_stream)    
        model_prompter.update_template(outputs)
        count += 1
