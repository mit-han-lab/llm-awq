import os
import torch
import gc


def auto_parallel(args):
    model_size = args.model_path.split("-")[-1]
    if model_size.endswith("m"):
        model_gb = 1
    else:
        model_gb = float(model_size[:-1])
    if model_gb < 20:
        n_gpu = 1
    elif model_gb < 50:
        n_gpu = 4
    else:
        n_gpu = 8
    args.parallel = n_gpu > 1
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if isinstance(cuda_visible_devices, str):
        cuda_visible_devices = cuda_visible_devices.split(",")
    else:
        cuda_visible_devices = list(range(8))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(dev) for dev in cuda_visible_devices[:n_gpu]]
    )
    print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    return cuda_visible_devices
