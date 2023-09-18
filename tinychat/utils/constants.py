import torch


def init():
    global max_seq_len, max_batch_size, llama_multiple_of, mem_efficient_load
    max_seq_len = 2048
    max_batch_size = 1
    llama_multiple_of = 256
    mem_efficient_load = False  # Whether to load the checkpoint in a layer-wise manner. Activate this if you are facing OOM issues on edge devices (e.g., Jetson Orin).
