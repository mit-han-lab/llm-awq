import gc
import os
import re
from typing import Union, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.quantize.quantizer import real_quantize_model_weight
from awq.quantize.qmodule import WQLinear
from tqdm import tqdm

import tinychat.utils.constants

version_message = """
[Warning] The awq quantized checkpoint seems to be in v1 format. 
If the model cannot be loaded successfully, please use the latest awq library to re-quantized the model, or repack the current checkpoint with tinychat/offline-weight-repacker.py
"""


def ckpt_version_check(quant_path):
    if not quant_path.endswith("v2.pt"):
        print(version_message)


def mem_efficient_load_checkpoint(
    model: nn.Module,
    ckpts_folder: Union[str, os.PathLike],
):
    checkpoint_files = [
        ckpts_folder + "/" + f for f in os.listdir(ckpts_folder) if f.endswith(".pt")
    ]

    # Check if the ckpts match the model
    model_keys = sorted((list(model.state_dict().keys())))
    suffix = r"\.pt$"
    ckpt_keys = sorted(
        [re.sub(suffix, "", f) for f in os.listdir(ckpts_folder) if f.endswith(".pt")]
    )
    assert len(model_keys) == len(
        ckpt_keys
    ), f"The number of checkpoint files do not match the model. \n Model has {len(model_keys)} keys, while finding {len(ckpt_keys)} checkpoint files in the folder."
    for key1, key2 in zip(model_keys, ckpt_keys):
        assert (
            key1 == key2
        ), f"The checkpoint files do not match the model. \nmodel key {key1} != checkpoint key {key2}"

    with tqdm(total=len(checkpoint_files)) as pbar:
        pbar.set_description("Loading checkpoint shards")
        for checkpoint_file in checkpoint_files:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint, strict=False)
            # Force Python to clean up.
            del checkpoint
            gc.collect()
            pbar.update(1)
    return model


def load_awq_model(model, checkpoint, w_bit, group_size, device):
    q_config = {"zero_point": True, "q_group_size": group_size}
    real_quantize_model_weight(model, w_bit, q_config, init_only=True)

    if hasattr(model.config, "tie_encoder_decoder"):
        model.config.tie_encoder_decoder = False
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    if tinychat.utils.constants.mem_efficient_load:
        assert os.path.isdir(
            checkpoint
        ), "You are in mem_efficient_load mode. \n Please set --load_quant the path to the folder containing all checkpoint files."
        model = mem_efficient_load_checkpoint(
            model,
            checkpoint,
        ).to(device)
    else:
        ckpt_version_check(checkpoint)
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint")
        for i in pbar:
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint,
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "LlamaDecoderLayer",
                    "BloomBlock",
                    "MPTBlock",
                    "DecoderLayer",
                    "CLIPEncoderLayer",
                ],
            ).to(device)
    return model


def make_quant_linear(module, names, w_bit, groupsize, device, name=""):
    if isinstance(module, WQLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            delattr(module, attr)
            setattr(
                module,
                attr,
                WQLinear(
                    w_bit,
                    groupsize,
                    tmp.in_features,
                    tmp.out_features,
                    tmp.bias is not None,
                    device,
                ),
            )
    for name1, child in module.named_children():
        make_quant_linear(
            child,
            names,
            w_bit,
            groupsize,
            device,
            name + "." + name1 if name != "" else name1,
        )


def find_layers(module, layers=[nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def load_awq_llama_fast(model, checkpoint, w_bit, group_size, device):
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant_linear(model, layers, w_bit, group_size, device)
    del layers

    if tinychat.utils.constants.mem_efficient_load:
        # TODO: mem-efficient load for llama
        assert os.path.isdir(
            checkpoint
        ), "You are in mem_efficient_load mode. \n Please set --load_quant the path to the folder containing all checkpoint files."
        model = mem_efficient_load_checkpoint(
            model,
            checkpoint,
        )
    else:
        ckpt_version_check(checkpoint)
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint")
        for i in pbar:
            if checkpoint.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load

                model.load_state_dict(safe_load(checkpoint))
            else:
                model.load_state_dict(torch.load(checkpoint))

    return model.to(device)
