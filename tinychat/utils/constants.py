import torch


def init():
    global max_seq_len, max_batch_size, llama_multiple_of, mem_efficient_load
    max_seq_len = 5120
    max_batch_size = 1
    llama_multiple_of = 256
    mem_efficient_load = False  # Whether to load the checkpoint in a layer-wise manner. Activate this if you are facing OOM issues on edge devices (e.g., Jetson Orin).

    # LLaVA Constants
    global LLAVA_IGNORE_INDEX, LLAVA_DEFAULT_IMAGE_TOKEN, LLAVA_DEFAULT_IMAGE_TOKEN_IDX, LLAVA_DEFAULT_IMAGE_PATCH_TOKEN, LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX, LLAVA_DEFAULT_IM_START_TOKEN, LLAVA_DEFAULT_IM_END_TOKEN, LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER, AUTO_FILL_IM_TOKEN_HOLDER
    LLAVA_IGNORE_INDEX = -100
    LLAVA_DEFAULT_IMAGE_TOKEN = "<image>"
    LLAVA_DEFAULT_IMAGE_TOKEN_IDX = -200
    LLAVA_DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX = 32000
    LLAVA_DEFAULT_IM_START_TOKEN = "<im_start>"
    LLAVA_DEFAULT_IM_END_TOKEN = "<im_end>"
    LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER = "<image>"
    AUTO_FILL_IM_TOKEN_HOLDER = "<im_holder>"

    # gradio UI
    global CONTROLLER_HEART_BEAT_EXPIRATION, WORKER_HEART_BEAT_INTERVAL
    CONTROLLER_HEART_BEAT_EXPIRATION = 30
    WORKER_HEART_BEAT_INTERVAL = 15
