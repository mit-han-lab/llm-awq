import argparse
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch

from tinychat.utils.tune import (
    device_warmup,
    tune_all_wqlinears,
    tune_llava_patch_embedding,
)
from tinychat.utils.prompt_templates import get_prompter, get_stop_token_ids
from tinychat.utils.llava_image_processing import process_images, load_image
from tinychat.models.llava_llama import LlavaLlamaForCausalLM
from tinychat.stream_generators.llava_stream_gen import LlavaStreamGenerator

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from attributedict.collections import AttributeDict

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
        ("temp", 0.20),  # 1.0 = disabled
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
        # print(f"Context Stage Time   : {context_time * 1000:.2f} ms")
        # print(f"Context Stage Tokens : {context_tokens} tokens")
        # print(f"Context Stage    : {context_time/context_tokens * 1000:.2f} ms/token")
        print(
            f"Generation Stage : {np.average(generation_time_list) * 1000:.2f} ms/token"
        )
        # print(f"Average Speed    : {average_speed * 1000:.2f} ms/token")
        print("=" * 50)
        # print("token num:", total_tokens)
        # print("Model total Time = ", (context_time + np.sum(generation_time_list))*1000, "ms" )
    return " ".join(output_text)


def skip(*args, **kwargs):
    pass


def main(args):
    # Accelerate model initialization
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = LlavaLlamaForCausalLM(config, args.device).half()
    vision_tower = model.get_model().vision_tower
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor
    vision_tower = vision_tower.half()

    if args.precision == "W16A16":
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint shards")
        for i in pbar:
            model = load_checkpoint_and_dispatch(
                model,
                args.model_path,
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "LlamaDecoderLayer",
                    "BloomBlock",
                    "MPTBlock",
                    "DecoderLayer",
                    "CLIPEncoderLayer",
                ],
            ).to(args.device)

    elif args.precision == "W4A16":
        from tinychat.utils.load_quant import load_awq_model

        model = load_awq_model(model, args.quant_path, 4, 128, args.device)
        from tinychat.modules import (
            make_quant_norm,
            make_quant_attn,
            make_fused_mlp,
            make_fused_vision_attn,
        )

        make_quant_attn(model, args.device)
        make_quant_norm(model)
        make_fused_mlp(model)
        # make_fused_vision_attn(model,args.device)
        model = model.to(args.device)

    else:
        raise NotImplementedError(f"Precision {args.precision} is not supported.")

    image = load_image(args.image_file)
    if args.vis_image:
        print("=" * 50)
        print("Input Image:")
        os.system(f"termvisage --query-timeout 1 {args.image_file}")
        print("=" * 50)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [
            image.to(args.device, dtype=torch.float16) for image in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(args.device, dtype=torch.float16)

    device_warmup(args.device)
    tune_llava_patch_embedding(vision_tower, device=args.device)

    stream_generator = LlavaStreamGenerator

    if args.max_seq_len <= 1024:
        short_prompt = True
    else:
        short_prompt = False
    model_prompter = get_prompter(args.model_type, args.model_path, short_prompt)
    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)
    count = 0

    model.eval()
    while True:
        # Get input from the user
        input_prompt = input("USER: ")
        if input_prompt == "":
            print("EXIT...")
            break
        if count == 0:  # Insert image here
            model_prompter.insert_prompt("<image>\n" + input_prompt)
        else:
            model_prompter.insert_prompt(input_prompt)
        output_stream = stream_generator(
            model,
            tokenizer,
            model_prompter.model_input,
            gen_params,
            device=args.device,
            stop_token_ids=stop_token_ids,
            image_tensor=image_tensor,
        )
        outputs = stream_output(output_stream)
        if (
            args.single_round is not True and args.max_seq_len > 512
        ):  # Only memorize previous conversations when kv_cache_size > 512
            model_prompter.update_template(outputs)
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="LLaMa", help="type of the model"
    )
    parser.add_argument(
        "--model-path", type=str, default="/data/llm/checkpoints/llava/llava-v1.5-7b"
    )
    parser.add_argument(
        "--quant-path",
        type=str,
        default="/data/llm/checkpoints/llava/llava-v1.5-7b-w4-g128-awq.pt",
    )
    parser.add_argument(
        "--precision", type=str, default="W4A16", help="compute precision"
    )
    parser.add_argument(
        "--image-file",
        type=str,
        default="https://llava.hliu.cc/file=/nobackup/haotian/code/LLaVA/llava/serve/examples/extreme_ironing.jpg",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--single_round",
        action="store_true",
        help="whether to memorize previous conversations",
    )
    parser.add_argument(
        "--vis-image",
        action="store_true",
        help="whether to visualize the image while chatting",
    )
    args = parser.parse_args()
    main(args)
