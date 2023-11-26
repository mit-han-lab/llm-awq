# Usage:
# Please first install awq/kernels
# then directly run CUDA_VISIBLE_DEVICES=0 python benchmark.py
import argparse
import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils
import tinychat.utils.constants
from tinychat.utils.load_quant import load_awq_model
from awq.quantize.quantizer import real_quantize_model_weight
from tinychat.utils.tune import tune_all_wqlinears, device_warmup
from tinychat.modules import make_quant_norm, make_quant_attn, make_fused_mlp


def skip(*args, **kwargs):
    pass


def main():
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
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Wheter to print more information.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="maximum sequence length for kv cache",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="maximum batch size for kv cache"
    )
    parser.add_argument(
        "--input_len", type=int, default=4, help="input context length"
    )
    parser.add_argument(
        "--output_len", type=int, default=200, help="generation token number"
    )
    parser.add_argument(
        "--precision", type=str, default="W4A16", help="compute precision"
    )
    args = parser.parse_args()

    tinychat.utils.constants.max_batch_size = args.batch_size
    tinychat.utils.constants.max_seq_len = args.max_seq_len
    from tinychat.models import (
        FalconForCausalLM,
        LlamaForCausalLM,
        MPTForCausalLM,
        GPTBigCodeForCausalLM,
        GPTNeoXForCausalLM,
    )

    modeling_utils._init_weights = False
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    device = "cuda:0"
    # exLLaMA benchmarking parameters.
    context_length = args.input_len
    gen_length = args.output_len
    input_ids = [1 for _ in range(context_length)]
    input_ids = torch.as_tensor([input_ids], device=device).repeat((args.batch_size, 1))

    model_type_dict = {
        "llama": LlamaForCausalLM,
        "falcon": FalconForCausalLM,
        "mpt": MPTForCausalLM,
        "mistral": LlamaForCausalLM,
        "starcoder": GPTBigCodeForCausalLM,
        "gptneox": GPTNeoXForCausalLM,
        "stablelm": LlamaForCausalLM,
    }

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    assert args.model_type.lower() in [
        "llama",
        "falcon",
        "mpt",
        "mistral",
        "starcoder",
        "stablelm",
        "gptneox",
    ], "We only support llama & falcon & mpt & mistral & starcoder & stablelm now"
    model = model_type_dict[args.model_type.lower()](config).half()

    if args.precision == "W4A16":
        real_quantize_model_weight(
            model,
            w_bit=4,
            q_config=dict(q_group_size=args.q_group_size, zero_point=True),
            init_only=True,
        )
    model = model.to(device)

    if args.precision == "W4A16":
        # tune_all_wqlinears(model)
        make_quant_attn(model, device)
        make_fused_mlp(model)
        make_quant_norm(model)
    device_warmup(device)

    print("huggingface ckpt loaded")
    print(model)

    time_lis = []

    start_pos = 0

    print("Benchmarking...")
    with torch.inference_mode():
        for i in range(gen_length):
            torch.cuda.synchronize()
            t_st = time.time()

            if i == 0:
                inputs = input_ids
            else:
                inputs = token
            out = model(inputs, start_pos=start_pos)
            start_pos += out.shape[1]

            torch.cuda.synchronize()
            t_ed = time.time()
            time_lis.append(t_ed - t_st)
            token = out[:, -1].max(1)[1].unsqueeze(1)

            if args.verbose:
                print(i, np.median(time_lis))

    print(f"Speed: {1 / np.median(time_lis) * args.batch_size} tokens per second.")


if __name__ == "__main__":
    main()
