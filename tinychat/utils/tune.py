import numpy as np
import time
import torch
from awq.quantize.qmodule import WQLinear


__all__ = ["device_warmup", "tune_all_wqlinears"]


def device_warmup(device: str):
    warm_up = torch.randn((8192, 8192)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)


def tune_llava_patch_embedding(vision_tower, device):
    # run the llava_patch_embedding layer to pre-tune the kernel configuration
    # Without this pre-tuning, the embedding layer can cause significant slowdown due to cuDNN tuning.
    device = vision_tower.device
    if "intern" not in vision_tower.__class__.__name__.lower():
        patch_embedding = (
            vision_tower.vision_tower.vision_model.embeddings.patch_embedding
        )
    else:
        patch_embedding = vision_tower.vision_tower.embeddings.patch_embedding
    patch_embedding = patch_embedding.to(device)
    image = (
        torch.randn((1, patch_embedding.in_channels, 336, 336))
        .to(device)
        .to(patch_embedding.weight.dtype)
    )
    for i in range(100):
        patch_embedding(image)


def _time_module(module, inputs, measure_iters=1000):
    time_lis = []
    # Warmup
    for i in range(measure_iters):
        module(inputs)
    for i in range(measure_iters):
        torch.cuda.synchronize()
        st = time.time()
        module(inputs)
        torch.cuda.synchronize()
        ed = time.time()
        time_lis.append((ed - st))
    return np.median(time_lis)


def tune_wqlinear(module: WQLinear, measure_iters: int = 1000):
    device_warmup(str(module.scales.device))
    inputs = torch.randn(
        1, module.in_features, device=module.scales.device, dtype=module.scales.dtype
    )
    best_split_k_iter = None
    best_latency = None
    for split_k_iters in [1, 2, 4, 8, 16, 32]:
        module.split_k_iters = split_k_iters
        cur_latency = _time_module(module, inputs, measure_iters)
        if best_split_k_iter is None or best_latency >= cur_latency:
            best_split_k_iter = split_k_iters
            best_latency = cur_latency
    module.split_k_iters = best_split_k_iter
    return best_split_k_iter


def tune_all_wqlinears(model, measure_iters: int = 1000):
    tuned_results = dict()
    for name, module in model.named_modules():
        if isinstance(module, WQLinear):
            ic, oc = module.in_features, module.out_features
            if (ic, oc) not in tuned_results:
                print(f"Tuning {(ic, oc)}...")
                split_k_iters = tune_wqlinear(module)
                tuned_results[(ic, oc)] = split_k_iters
    # write configs to model
    for name, module in model.named_modules():
        if isinstance(module, WQLinear):
            ic, oc = module.in_features, module.out_features
            module.split_k_iters = tuned_results[(ic, oc)]
