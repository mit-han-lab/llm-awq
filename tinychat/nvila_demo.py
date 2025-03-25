import argparse

from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
import torch
from awq.quantize import fake_quant
from transformers import AutoConfig
from tinychat.utils.load_quant import load_awq_model
from tinychat.utils.llava_image_processing import (
    load_images,
    vis_images,
)


def skip(*args, **kwargs):
    pass


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
from llava.utils.media import extract_media
import tinychat.utils.constants
from tinychat.stream_generators.NVILA_stream_gen import NVILAStreamGenerator
from tinychat.utils.conversation_utils import gen_params, stream_output, TimeStats

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    # Accelerate model initialization
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    tinychat.utils.constants.max_seq_len = args.max_seq_len

    # Prepare model
    from tinychat.models.nvila_qwen2 import NVILAQwen2
    from transformers import AutoConfig
    from tinychat.models.qwen2 import Qwen2ForCausalLM

    config = AutoConfig.from_pretrained(args.model_path)
    config.resume_path = args.model_path
    if args.quant_llm or args.all:
        model = NVILAQwen2(config, False).half()
    else:
        model = NVILAQwen2(config, True).half()

    if args.smooth_VT or args.all:
        from awq.quantize import smooth_lm

        act_scales = torch.load(args.act_scale_path)
        smooth_lm(model.vision_tower, act_scales, 0.3)
    if args.quant_llm or args.all:
        from tinychat.modules import (
            make_quant_norm,
            make_quant_attn,
            make_fused_mlp,
            make_fused_vision_attn,
        )

        model.llm = Qwen2ForCausalLM(model.llm_cfg).half()
        model.llm = load_awq_model(model.llm, args.quant_path, 4, 128, args.device)
        make_quant_attn(model.llm, args.device, True)
        make_quant_norm(model.llm)
        model.llm.cpu()
        model.llm.resize_token_embeddings(len(model.tokenizer))

    if args.quant_VT or args.all:
        from tinychat.modules import QuantSiglipEncoder

        if args.fakequant_VT:
            fake_quant(model.vision_tower.vision_tower.vision_model.encoder)
        else:
            model.vision_tower.vision_tower.vision_model.encoder = QuantSiglipEncoder(
                model.vision_tower.vision_tower.vision_model.encoder
            )
    model = model.cuda().eval()
    device_warmup(args.device)
    tune_llava_patch_embedding(model.vision_tower, device=args.device)

    # Pre-prepare media
    prompt = []
    media_files = []
    if args.media is not None:
        for media in args.media or []:
            if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                media = Image(media)
                media_files.append(media)
                media_prompt = "<image>"
            elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                media = Video(media)
                media_files.append(media)
                media_prompt = "<vila/video>"
            else:
                raise ValueError(f"Unsupported media type: {media}")
            prompt.append(media)
    media_num = len(media_files)
    if args.vis_image:
        print("=" * 50)
        print("Input Image:")
        vis_images(args.media)
    conversation = [{"from": "human", "value": prompt}]
    media, media_cfg = model.prepare_media(conversation)
    # Prepare streaming
    stream_generator = NVILAStreamGenerator
    # Prepare prompt
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

    count = 0
    model.eval()
    time_stats = TimeStats()
    start_pos = 0
    while True:
        # Get input from the user
        print("=" * 50)
        input_prompt = input(input_indicator)
        print("-" * 50)
        if input_prompt == "":
            print("EXIT...")
            time_stats.show()
            break
        if count == 0:  # Insert media here
            if args.media is not None:
                if media_prompt in input_prompt:
                    input_prompt = input_prompt
                else:
                    input_prompt = media_prompt * media_num + input_prompt
            model_prompter.insert_prompt(input_prompt)
        else:
            model_prompter.insert_prompt(input_prompt)
            if args.chunk_prefilling:
                media = None
                media_cfg = None
        output_stream = stream_generator(
            model,
            gen_params,
            model_prompter.model_input,
            media,
            media_cfg,
            start_pos,
            device=args.device,
            stop_token_ids=stop_token_ids,
            chunk_prefilling=args.chunk_prefilling,
            quant_llm=args.quant_llm or args.all,
        )
        print(output_indicator, end="", flush=True)
        if count == 0:
            outputs, total_tokens = stream_output(output_stream, time_stats)
        else:
            outputs, total_tokens = stream_output(output_stream)
        if args.chunk_prefilling:
            start_pos += total_tokens
        if (
            args.single_round is not True and args.max_seq_len > 512
        ):  # Only memorize previous conversations when kv_cache_size > 512
            model_prompter.update_template(outputs, args.chunk_prefilling)
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
        "--quant_path",
        type=str,
        default="/data/llm/checkpoints/llava/llava-v1.5-7b-w4-g128-awq.pt",
    )
    parser.add_argument(
        "--act_scale_path",
        type=str,
        default="/PATH/TO/SCALE",
    )
    parser.add_argument(
        "--media", type=str, nargs="+", help="Multi-modal input (Video or image path)"
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
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="whether to use flash attention",
    )
    parser.add_argument(
        "--chunk_prefilling",
        action="store_true",
        help="If used, in context stage, the history tokens will not be recalculated, greatly speeding up the calculation",
    )
    # smooth and quantization options
    parser.add_argument("--quant_llm", action="store_true")
    parser.add_argument("--quant_VT", action="store_true")
    parser.add_argument("--smooth_VT", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--fakequant_VT",
        action="store_true",
        help="Use fake quant or real quant for VisionTower",
    )
    args = parser.parse_args()
    main(args)
