import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils
from attributedict.collections import AttributeDict
from tinychat.stream_generators import StreamGenerator
import tinychat.utils.constants
from tinychat.utils.load_quant import load_awq_model, load_awq_llama_fast
from tinychat.utils.prompt_templates import get_prompter, get_stop_token_ids
from tinychat.utils.tune import device_warmup, tune_all_wqlinears

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# opt_params in TinyLLMEngine
gen_params = AttributeDict(
    [
        ("seed", -1),  # RNG seed
        ("n_threads", 1),  # TODO: fix this
        ("n_predict", 512),  # new tokens to predict
        ("n_parts", -1),  # amount of model parts (-1: determine from model dimensions)
        ("n_ctx", 512),  # context size
        ("n_batch", 512),  # batch size for prompt processing (must be >=32 to use BLAS)
        ("n_keep", 0),  # number of tokens to keep from initial prompt
        ("n_vocab", 50272),  # vocabulary size
        # sampling parameters
        ("logit_bias", dict()),  # logit bias for specific tokens: <int, float>
        ("top_k", 40),  # <= 0 to use vocab size
        ("top_p", 0.95),  # 1.0 = disabled
        ("tfs_z", 1.00),  # 1.0 = disabled
        ("typical_p", 1.00),  # 1.0 = disabled
        ("temp", 0.70),  # 1.0 = disabled
        ("repeat_penalty", 1.10),  # 1.0 = disabled
        (
            "repeat_last_n",
            64,
        ),  # last n tokens to penalize (0 = disable penalty, -1 = context size)
        ("frequency_penalty", 0.00),  # 0.0 = disabled
        ("presence_penalty", 0.00),  # 0.0 = disabled
        ("mirostat", 0),  # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        ("mirostat_tau", 5.00),  # target entropy
        ("mirostat_eta", 0.10),  # learning rate
    ]
)


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
        average_speed = (context_time + np.sum(generation_time_list)) / (
            context_tokens + generation_tokens
        )
        print("=" * 50)
        print("Speed of Inference")
        print("-" * 50)
        print(
            f"Generation Stage : {np.average(generation_time_list) * 1000:.2f} ms/token"
        )
        print("=" * 50)
    return " ".join(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="LLaMa", help="type of the model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/llm/checkpoints/vicuna-hf/vicuna-7b",
        help="path to the model",
    )
    parser.add_argument(
        "--precision", type=str, default="W4A16", help="compute precision"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument(
        "--load_quant",
        type=str,
        default="/data/llm/checkpoints/vicuna-hf/vicuna-7b-awq-w4g128.pt",
        help="path to the pre-quanted 4-bit weights",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="maximum sequence length for kv cache",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=1, help="maximum batch size for kv cache"
    )
    parser.add_argument(
        "--mem_efficient_load",
        action="store_true",
        help="enable mem_efficient_load mod",
    )
    parser.add_argument(
        "--single_round",
        action="store_true",
        help="whether to memorize previous conversations",
    )

    args = parser.parse_args()
    assert args.model_type.lower() in [
        "llama",
        "falcon",
        "mpt",
    ], "We only support llama & falcon & mpt now"
    assert args.precision in ["W4A16", "W16A16"], "We only support W4A16/W16A16 now"

    gen_params.n_predict = 512
    gen_params.n_vocab = 32000
    tinychat.utils.constants.max_batch_size = args.max_batch_size
    tinychat.utils.constants.max_seq_len = args.max_seq_len
    tinychat.utils.constants.mem_efficient_load = args.mem_efficient_load
    if tinychat.utils.constants.mem_efficient_load:
        print("=" * 80)
        print(
            "[Info] You have activated mem_efficient_load mode.\n       Less on-chip memory will be consumed when loading the model.\n       However, the loading process will take more time."
        )
        print("=" * 80)
    # TODO (Haotian): a more elegant implementation here.
    # We need to update these global variables before models use them.
    from tinychat.models import FalconForCausalLM, LlamaForCausalLM, MPTForCausalLM

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if "mpt" in config.__class__.__name__.lower():
        # config.init_device="meta"
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=False, trust_remote_code=True
        )
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)

    model_type_dict = {
        "llama": LlamaForCausalLM,
        "falcon": FalconForCausalLM,
        "mpt": MPTForCausalLM,
    }

    if args.precision == "W4A16":
        if args.model_type.lower() == "llama":
            model = model_type_dict["llama"](config).half()
            model = load_awq_llama_fast(
                model, args.load_quant, 4, args.q_group_size, args.device
            )
        else:
            model = model_type_dict[args.model_type.lower()](config).half()
            model = load_awq_model(
                model, args.load_quant, 4, args.q_group_size, args.device
            )
    else:
        loaded_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = model_type_dict[args.model_type.lower()](config).half().to(args.device)
        model.load_state_dict(loaded_model.state_dict())

    # device warm up
    device_warmup(args.device)
    # autotune split_k_iters
    # tune_all_wqlinears(model)

    # TODO (Haotian): Verify if the StreamGenerator still works for the unmodified falcon impl.
    stream_generator = StreamGenerator

    # Optimize AWQ quantized model
    if args.precision == "W4A16" and args.model_type.lower() == "llama":
        from tinychat.modules import make_quant_norm, make_quant_attn

        make_quant_attn(model, args.device)
        make_quant_norm(model)

    if args.max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False
    model_prompter = get_prompter(args.model_type, args.model_path, short_prompt)
    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)
    count = 0
    while True:
        # Get input from the user
        input_prompt = input("USER: ")
        if input_prompt == "":
            print("EXIT...")
            break
        model_prompter.insert_prompt(input_prompt)
        output_stream = stream_generator(
            model,
            tokenizer,
            model_prompter.model_input,
            gen_params,
            device=args.device,
            stop_token_ids=stop_token_ids,
        )
        outputs = stream_output(output_stream)
        if (
            args.single_round is not True and args.max_seq_len > 512
        ):  # Only memorize previous conversations when kv_cache_size > 512
            model_prompter.update_template(outputs)
        count += 1
