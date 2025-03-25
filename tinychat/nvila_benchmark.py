import argparse

from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
import torch
from awq.quantize import fake_quant
from awq.quantize.quantizer import real_quantize_model_weight
from transformers import AutoConfig
import tinychat


def skip(*args, **kwargs):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="/home/yuming/workspace/qwen/models/nvila-internal-8b-v1",
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default="/PATH/TO/QUANT",
    )
    # parser.add_argument("--model-path", "-m", type=str, default="Efficient-Large-Model/J65")
    # parser.add_argument("--quant_path", type=str, default="/home/yuming/workspace/qwen/models/J65/llm/vila2-J65-w4-g128-awq-v2.pt")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    # parser.add_argument("--media", type=str, default="/home/yuming/workspace/space_woaudio.mp4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--act_scale_path",
        type=str,
        default="/PATH/TO/SCALE",
    )
    # quantization options
    parser.add_argument("--quant_llm", action="store_true")
    parser.add_argument("--quant_VT", action="store_true")
    # Four basic tasks
    parser.add_argument("--video_caption", action="store_true")
    parser.add_argument("--video_QA", action="store_true")
    parser.add_argument("--image_caption", action="store_true")
    parser.add_argument("--image_QA", action="store_true")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Whether to quantize visiontower and llm, and test all 4 tasks",
    )
    parser.add_argument(
        "--fakequant_VT",
        action="store_true",
        help="Use fake quant or real quant for VisionTower",
    )
    parser.add_argument(
        "--all_task", action="store_true", help="Whether to test all 4 tasks"
    )
    parser.add_argument(
        "--video_path", type=str, default="../figures/nvila_demo_video.mp4"
    )
    parser.add_argument("--image_path", type=str, default="../figures/vila-logo.jpg")
    parser.add_argument("--max_seq_len", type=int, default=8192)
    args = parser.parse_args()

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    import tinychat.utils.constants

    tinychat.utils.constants.max_seq_len = args.max_seq_len
    from transformers import modeling_utils

    modeling_utils._init_weights = False

    # Load model
    from tinychat.models.nvila_qwen2 import NVILAQwen2

    config = AutoConfig.from_pretrained(args.model_path)
    config.resume_path = args.model_path
    model = NVILAQwen2(config).half()
    model.llm = model.llm.eval()
    if args.quant_llm or args.all:
        from tinychat.modules import (
            make_quant_norm,
            make_quant_attn,
            make_fused_mlp,
            make_fused_vision_attn,
        )

        real_quantize_model_weight(
            model.llm,
            w_bit=4,
            q_config=dict(q_group_size=128, zero_point=True),
            init_only=True,
        )
        make_quant_attn(model.llm, "cuda", True)
        make_quant_norm(model.llm)
        make_fused_mlp(model.llm)
        model = model.to("cuda")
    model = model.to(args.device)
    if args.quant_VT or args.all:
        from tinychat.modules import QuantSiglipEncoder

        model.vision_tower.vision_tower.vision_model.encoder = QuantSiglipEncoder(
            model.vision_tower.vision_tower.vision_model.encoder
        )
    model = model.cuda().eval()

    if args.video_caption or args.all or args.all_task:
        print("-" * 80)
        print("Video_Caption")
        # Set conversation mode
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
        media = Video(args.video_path)
        text = "Elaborate on the visual and narrative elements of the video in detail."  # + "1"+" 1"*3069
        prompt = [media, text]
        # Generate response
        with torch.no_grad():
            response = model.benchmark(prompt, args.quant_llm)
    if args.video_QA or args.all or args.all_task:
        print("-" * 80)
        print("Video_QA")
        # Set conversation mode
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
        media = Video(args.video_path)
        text = "What is the person in the video doing? Select the option that best describes their action: A. Folding paper B. Playing computer games C. Sleeping."  # + "1"+" 1"*3069
        prompt = [media, text]
        # Generate response
        with torch.no_grad():
            response = model.benchmark(prompt, args.quant_llm)
    if args.image_caption or args.all or args.all_task:
        print("-" * 80)
        print("Image_Caption")
        # Set conversation mode
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
        media = Image(args.image_path)
        text = "Describe the image in detail."
        prompt = [media, text]
        # Generate response
        with torch.no_grad():
            response = model.benchmark(prompt, args.quant_llm)
    if args.image_QA or args.all or args.all_task:
        print("-" * 80)
        print("Image_QA")
        # Set conversation mode
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
        media = Image(args.image_path)
        text = "What does the text in the image say? Choose the option that best matches: A. VILA B. AIIV C. ALIV."
        prompt = [media, text]
        # Generate response
        with torch.no_grad():
            response = model.benchmark(prompt, args.quant_llm)


if __name__ == "__main__":
    main()
