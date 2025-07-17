# Adapted from SmoothQuant (https://github.com/mit-han-lab/smoothquant) and modified by Yuming Lou


import torch.nn as nn
try:
    import llava
    from llava.media import Image, Video
    from llava.utils.media import extract_media
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import process_image, process_images
except ImportError:
    print("VILA is not installed. Multimodal features will not be available. To activate, please install VILA at https://github.com/NVlabs/VILA.")

import torch
from collections import defaultdict
from functools import partial
from tqdm import tqdm
import numpy as np
import functools


@torch.no_grad()
def get_act_scales(model, data):
    num_samples = data.shape[0]
    model.eval()
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        input = data[i : i + 1]
        model(input)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    data,
):
    num_samples = data.shape[1]
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))
    pbar = tqdm(range(num_samples))
    for i in pbar:
        model(data[i : i + 1])
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()
    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = (
            act_dict[
                f"vision_tower.vision_model.encoder.layers.{idx}.self_attn.q_proj"
            ]["input"]
            / 127
        )
        scale_dict["q_output_scale"] = (
            act_dict[
                f"vision_tower.vision_model.encoder.layers.{idx}.self_attn.q_proj"
            ]["output"]
            / 127
        )
        scale_dict["k_output_scale"] = (
            act_dict[
                f"vision_tower.vision_model.encoder.layers.{idx}.self_attn.k_proj"
            ]["output"]
            / 127
        )
        scale_dict["v_output_scale"] = (
            act_dict[
                f"vision_tower.vision_model.encoder.layers.{idx}.self_attn.v_proj"
            ]["output"]
            / 127
        )
        scale_dict["out_input_scale"] = (
            act_dict[
                f"vision_tower.vision_model.encoder.layers.{idx}.self_attn.out_proj"
            ]["input"]
            / 127
        )
        scale_dict["fc1_input_scale"] = (
            act_dict[f"vision_tower.vision_model.encoder.layers.{idx}.mlp.fc1"]["input"]
            / 127
        )
        scale_dict["fc2_input_scale"] = (
            act_dict[f"vision_tower.vision_model.encoder.layers.{idx}.mlp.fc2"]["input"]
            / 127
        )
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict


def get_smooth_scale(model_path, media):
    # Load model
    model = llava.load(model_path, devices=[0])
    del model.llm
    del model.mm_projector
    torch.cuda.empty_cache()
    model = model.cuda().eval()
    prompt = []
    if media is not None:
        for m in media or []:
            if any(m.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                m = Image(m)
            elif any(m.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                m = Video(m)
            else:
                raise ValueError(f"Unsupported media type: {m}")
            prompt.append(m)
    conversation = [{"from": "human", "value": prompt}]
    media = extract_media(conversation, model.config)
    for name in media:
        if name == "image":
            if (
                len(media["image"]) == 1
                and model.config.image_aspect_ratio == "dynamic"
            ):
                model.config.image_processor = model.vision_tower.image_processor
                images = process_image(
                    media["image"][0], model.config, None, enable_dynamic_res=True
                ).half()
                conversation[0]["value"] = conversation[0]["value"].replace(
                    DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                )
            else:
                images = process_images(
                    media["image"], model.vision_tower.image_processor, model.config
                ).half()
            media[name] = [image for image in images]
        elif name == "video":
            media[name] = [
                process_images(
                    images, model.vision_tower.image_processor, model.config
                ).half()
                for images in media[name]
            ]
        else:
            raise ValueError(f"Unsupported media type: {name}")
    images = torch.cat(media["video"], dim=1)
    model.vision_tower = model.vision_tower.eval()
    decoder_layer_scales = get_act_scales(model.vision_tower, images)
    return decoder_layer_scales


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    if "siglip" in str(model.__class__).lower():
        num = 0
        for name, module in model.named_modules():
            if "siglipencoderlayer" in str(module.__class__).lower():
                attn_ln = module.layer_norm1
                qkv = [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ]
                qkv_input_scales = scales[name + ".self_attn.q_proj"]
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

                ffn_ln = module.layer_norm2
                fc1 = module.mlp.fc1
                fc1_input_scales = scales[name + ".mlp.fc1"]
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
                num += 1
