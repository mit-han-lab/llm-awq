import argparse
import torch

from PIL import Image
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch

from tinychat.utils.tune import (
    device_warmup,
    tune_all_wqlinears,
    tune_llava_patch_embedding,
)
from tinychat.utils.prompt_templates import (
    get_prompter,
    get_stop_token_ids,
    get_image_token,
)
from tinychat.utils.llava_image_processing import (
    process_images,
    load_images,
    vis_images,
)
import tinychat.utils.constants
from tinychat.models.llava_llama import LlavaLlamaForCausalLM
from tinychat.stream_generators.llava_stream_gen import LlavaStreamGenerator
from tinychat.utils.conversation_utils import gen_params, stream_output, TimeStats

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def image_parser(args):
    out = args.image_file.split(args.im_sep)
    return out


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
    tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX = (
        tokenizer.convert_tokens_to_ids(
            [tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
    )
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.min_max_range_path = args.model_path + "/emb_min_max.pt"
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
        # make_fused_mlp(model)
        # make_fused_vision_attn(model,args.device)
        model = model.to(args.device)

    else:
        raise NotImplementedError(f"Precision {args.precision} is not supported.")

    image_files = image_parser(args)
    image_num = len(image_files)
    images = load_images(image_files)
    if args.vis_image:
        print("=" * 50)
        print("Input Image:")
        vis_images(image_files)
    # Similar operation in model_worker.py
    image_tensor = process_images(images, image_processor, model.config)
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
    model_prompter = get_prompter(
        args.model_type, args.model_path, short_prompt, args.empty_prompt
    )
    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)
    count = 0

    if args.empty_prompt:
        input_indicator = "Input: "
        output_indicator = "Generated: "
    else:
        input_indicator = "USER: "
        output_indicator = "ASSISTANT: "

    model.eval()
    time_stats = TimeStats()
    while True:
        # Get input from the user
        print("=" * 50)
        input_prompt = input(input_indicator)
        print("-" * 50)
        if input_prompt == "":
            print("EXIT...")
            time_stats.show()
            break
        if count == 0:  # Insert image here
            image_token = get_image_token(model, args.model_path)
            image_token_holder = (
                tinychat.utils.constants.LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER
            )
            im_token_count = input_prompt.count(image_token_holder)
            if im_token_count == 0:
                model_prompter.insert_prompt(image_token * image_num + input_prompt)
            else:
                assert im_token_count == image_num
                input_prompt = input_prompt.replace(image_token_holder, image_token)
                model_prompter.insert_prompt(input_prompt)
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
        print(output_indicator, end="", flush=True)
        if count == 0:
            outputs = stream_output(output_stream, time_stats)
        else:
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
    parser.add_argument(
        "--im-sep",
        type=str,
        default=",",
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
    parser.add_argument(
        "--empty-prompt",
        action="store_true",
        help="whether to use empty prompt template",
    )
    args = parser.parse_args()
    main(args)
