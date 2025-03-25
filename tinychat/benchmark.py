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
from tinychat.utils.tune import (
    tune_all_wqlinears,
    device_warmup,
    tune_llava_patch_embedding,
)
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
        default=8192,
        help="maximum sequence length for kv cache",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=1, help="maximum batch size for kv cache"
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
    parser.add_argument(
        "--context_length",
        type=list,
        nargs="+",
        help="The length of input. And if chunk_prefilling used, this serves as the length of tokens from history rounds.",
    )
    parser.add_argument(
        "--question_length",
        type=list,
        nargs="+",
        help="The length of new input. Only useful and necessary when benchmarking chunk_prefilling method",
    )
    parser.add_argument(
        "--precision", type=str, default="W4A16", help="compute precision"
    )
    args = parser.parse_args()
    # some checks
    assert (args.question_length is not None and args.chunk_prefilling) or (
        not args.chunk_prefilling
    ), "If you want to benchmark chunk prefilling, you need specify the question length and context length"
    assert args.precision in ["W4A16", "W16A16"], "We only support W4A16/W16A16 now"
    token_num = 256
    # We support fixing a certain kind of length
    if args.chunk_prefilling:
        if len(args.context_length) == 1 and len(args.question_length) > 1:
            args.context_length = [
                args.context_length[0] for _ in range(len(args.question_length))
            ]
        elif len(args.question_length) == 1 and len(args.context_length) > 1:
            args.question_length = [
                args.question_length[0] for _ in range(len(args.context_length))
            ]
        elif len(args.question_length) != len(args.context_length):
            raise ValueError(
                "The number of items in the question_length and context_length is expected to be either one or equal!"
            )
    tinychat.utils.constants.max_batch_size = args.max_batch_size
    tinychat.utils.constants.max_seq_len = args.max_seq_len
    from tinychat.models import FalconForCausalLM, LlamaForCausalLM, MPTForCausalLM
    from tinychat.models.vila_llama import VilaLlamaForCausalLM

    modeling_utils._init_weights = False
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    device = "cuda:0"
    model_type_dict = {
        "llama": LlamaForCausalLM,
        "falcon": FalconForCausalLM,
        "mpt": MPTForCausalLM,
    }

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    assert args.model_type.lower() in [
        "llama",
        "falcon",
        "mpt",
        "vila",
    ], "We only support llama & falcon & mpt & vila now"
    if "vila" in args.model_type.lower():
        model = VilaLlamaForCausalLM(config).half()
        print(model)
        if args.precision in ["W4A16"]:
            real_quantize_model_weight(
                model.llm,
                w_bit=4,
                q_config=dict(q_group_size=args.q_group_size, zero_point=True),
                init_only=True,
            )
            make_quant_attn(model.llm, device, args.flash_attn)
            make_quant_norm(model.llm)
            make_fused_mlp(model.llm)
        model = model.to(device)
        device_warmup(device)
        tune_llava_patch_embedding(model.get_vision_tower(), device=device)
        if not args.chunk_prefilling:
            image_num = [
                int(int("".join(i)) * 1 / 196) for i in args.context_length
            ]  # consider about three thirds of the history tokens are images
            if sum(image_num) > 0:
                image_tensor = 2 * torch.rand((max(image_num), 3, 384, 384)) - 1
                image_tensor = image_tensor.half().to(device)
            else:
                image_tensor = None

        print("huggingface ckpt loaded")

        # warming up
        input_ids = [1 for _ in range(2048)]
        inputs = torch.as_tensor([input_ids], device=device)
        out = model(
            inputs, start_pos=0, chunk_prefilling=args.chunk_prefilling
        )  # warmup

        if not args.chunk_prefilling:
            for i, context_length in enumerate(args.context_length):
                context_length = int("".join(context_length))
                time_lis = []
                if image_num[i]:
                    images = image_tensor[0 : image_num[i], :, :, :]
                    input_ids = [-200 for _ in range(image_num[i])] + [
                        1 for _ in range(context_length - 196 * image_num[i])
                    ]
                else:
                    images = None
                    input_ids = [1 for _ in range(context_length)]
                print("-" * 80)
                print(
                    "Context length: {} with {} pictures".format(
                        context_length, image_num[i]
                    )
                )
                with torch.inference_mode():
                    for i in range(10):  # Run ten times and get the average value
                        start_pos = 0
                        torch.cuda.synchronize()
                        t_st = time.time()
                        inputs = torch.as_tensor([input_ids], device=device)
                        out = model(
                            inputs,
                            start_pos=start_pos,
                            chunk_prefilling=args.chunk_prefilling,
                            images=images,
                        )
                        start_pos += inputs.shape[1]
                        torch.cuda.synchronize()
                        t_ed = time.time()
                        token = out[:, -1].max(1)[1].unsqueeze(1)
                        time_lis.append(t_ed - t_st)
                        if args.verbose:
                            print(i, t_ed - t_st)
                    print(f"Time To First Token: {np.mean(time_lis):.5f} s.")
                    print("-" * 80)
        else:
            for i, (context_length, question_length) in enumerate(
                zip(args.context_length, args.question_length)
            ):
                context_length = int("".join(context_length))
                question_length = int("".join(question_length))
                input_ids_old = [1 for _ in range(context_length)]
                images = None
                input_ids_new = [1 for _ in range(question_length)]
                time_lis = []
                print("-" * 80)
                print(
                    "History length: {} ; Question length: {}".format(
                        context_length, question_length
                    )
                )
                with torch.inference_mode():
                    for i in range(10):  # Run ten times and get the average value
                        # history rounds
                        start_pos = 0
                        if context_length > question_length:
                            inputs = torch.as_tensor([input_ids_old], device=device)
                            out = model(
                                inputs,
                                start_pos=start_pos,
                                chunk_prefilling=args.chunk_prefilling,
                                images=None,
                            )
                            start_pos += context_length

                        # the present round
                        torch.cuda.synchronize()
                        t_st = time.time()
                        inputs = torch.as_tensor([input_ids_new], device=device)
                        out = model(
                            inputs,
                            start_pos=start_pos,
                            chunk_prefilling=args.chunk_prefilling,
                        )
                        start_pos += inputs.shape[1]
                        torch.cuda.synchronize()
                        t_ed = time.time()

                        token = out[:, -1].max(1)[1].unsqueeze(1)
                        time_lis.append(t_ed - t_st)
                        if args.verbose:
                            print(i, t_ed - t_st)
                    print(
                        f"Time To First Token of this round: {np.mean(time_lis):.5f} s."
                    )
                    print("-" * 80)
    else:
        model = model_type_dict[args.model_type.lower()](config).half()
        if args.precision in ["W4A16"]:
            real_quantize_model_weight(
                model,
                w_bit=4,
                q_config=dict(q_group_size=args.q_group_size, zero_point=True),
                init_only=True,
            )
        model = model.to(device)

        if args.precision in ["W4A16"]:
            # tune_all_wqlinears(model)
            make_quant_attn(model, device, args.flash_attn)
            make_quant_norm(model)
            make_fused_mlp(model)
        device_warmup(device)

        print("huggingface ckpt loaded")

        # warming up
        input_ids = [1 for _ in range(2048)]
        inputs = torch.as_tensor([input_ids], device=device)
        out = model(
            inputs,
            start_pos=0,
            chunk_prefilling=args.chunk_prefilling,
            quant=args.precision in ["W4A16"],
        )  # warmup

        if not args.chunk_prefilling:
            for context_length in args.context_length:
                context_length = int("".join(context_length))
                input_ids = [1 for _ in range(context_length)]
                time_lis = []
                print("-" * 80)
                print("Context length: {}".format(context_length))
                with torch.inference_mode():
                    for i in range(10):  # Run ten times and get the average value
                        start_pos = 0
                        torch.cuda.synchronize()
                        t_st = time.time()
                        inputs = torch.as_tensor([input_ids], device=device)
                        out = model(
                            inputs,
                            start_pos=start_pos,
                            chunk_prefilling=args.chunk_prefilling,
                            quant=args.precision in ["W4A16"],
                        )
                        start_pos += inputs.shape[1]
                        torch.cuda.synchronize()
                        t_ed = time.time()
                        token = torch.argmax(out, keepdim=True)[0]
                        time_lis.append(t_ed - t_st)
                        if args.verbose:
                            print(i, t_ed - t_st)
                    print(f"Time To First Token: {np.mean(time_lis):.5f} s.")
                    # decoing throughput
                    time_lis = []
                    start_pos = context_length
                    torch.cuda.synchronize()
                    t_st = time.time()
                    for i in range(token_num):
                        token = model(
                            token,
                            start_pos=start_pos,
                            chunk_prefilling=args.chunk_prefilling,
                            quant=args.precision in ["W4A16"],
                        )
                        start_pos += 1
                        token = torch.argmax(token, keepdim=True)[0]
                        torch.cuda.synchronize()
                    t_ed = time.time()
                    time_lis.append(t_ed - t_st)
                    print(
                        f"Decoding throughput: {token_num/sum(time_lis):.5f} token/s."
                    )
                    print("-" * 80)
        else:
            for context_length, question_length in zip(
                args.context_length, args.question_length
            ):
                context_length = int("".join(context_length))
                question_length = int("".join(question_length))
                input_ids_old = [1 for _ in range(context_length)]
                input_ids_new = [1 for _ in range(question_length)]
                time_lis = []
                print("-" * 80)
                print(
                    "History length: {} ; Question length: {}".format(
                        context_length, question_length
                    )
                )
                with torch.inference_mode():
                    for i in range(10):  # Run ten times and get the average value
                        # history rounds
                        start_pos = 0
                        if context_length > question_length:
                            inputs = torch.as_tensor([input_ids_old], device=device)
                            out = model(
                                inputs,
                                start_pos=start_pos,
                                chunk_prefilling=args.chunk_prefilling,
                                quant=args.precision in ["W4A16"],
                            )
                            start_pos += inputs.shape[1]

                        # the present round
                        torch.cuda.synchronize()
                        t_st = time.time()
                        inputs = torch.as_tensor([input_ids_new], device=device)
                        out = model(
                            inputs,
                            start_pos=start_pos,
                            chunk_prefilling=args.chunk_prefilling,
                            quant=args.precision in ["W4A16"],
                        )
                        start_pos += inputs.shape[1]
                        torch.cuda.synchronize()
                        t_ed = time.time()

                        token = out[:, -1].max(1)[1].unsqueeze(1)
                        time_lis.append(t_ed - t_st)
                        if args.verbose:
                            print(i, t_ed - t_st)
                    print(
                        f"Time To First Token of this round: {np.mean(time_lis):.5f} s."
                    )
                    print("-" * 80)


if __name__ == "__main__":
    main()
