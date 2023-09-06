import torch

def init():
    global max_seq_len, max_batch_size, llama_multiple_of
    max_seq_len = 2048
    max_batch_size = 1
    llama_multiple_of = 256