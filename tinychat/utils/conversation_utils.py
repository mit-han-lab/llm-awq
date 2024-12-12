from typing import Dict
import numpy as np
from attributedict.collections import AttributeDict

gen_params = AttributeDict(
    [
        ("seed", -1),  # RNG seed
        ("n_threads", 1),  # TODO: fix this
        ("n_predict", 512),  # new tokens to predict
        ("n_parts", -1),  # amount of model parts (-1: determine from model dimensions)
        ("n_ctx", 512),  # context size
        ("n_batch", 512),  # batch size for prompt processing (must be >=32 to use BLAS)
        ("n_keep", 0),  # number of tokens to keep from initial prompt
        ("n_vocab", 50272),  # vocabulary size
        # sampling parameters
        ("logit_bias", dict()),  # logit bias for specific tokens: <int, float>
        ("top_k", 50),  # <= 0 to use vocab size
        ("top_p", 0.95),  # 1.0 = disabled
        ("tfs_z", 1.00),  # 1.0 = disabled
        ("typical_p", 1.00),  # 1.0 = disabled
        ("temp", 0.20),  # 1.0 = disabled
        ("repeat_penalty", 1.10),  # 1.0 = disabled
        (
            "repeat_last_n",
            64,
        ),  # last n tokens to penalize (0 = disable penalty, -1 = context size)
        ("frequency_penalty", 0.00),  # 0.0 = disabled
        ("presence_penalty", 0.00),  # 0.0 = disabled
        ("mirostat", 0),  # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        ("mirostat_tau", 5.00),  # target entropy
        ("mirostat_eta", 0.10),  # learning rate
    ]
)


class TimeStats:
    def __init__(self):
        self.total_tokens = 0
        self.context_tokens = 0
        self.context_time = 0.0

        self.generation_tokens = 0
        self.generation_time_list = []

    def update(self, timing: Dict):
        self.context_tokens = timing["context_tokens"]
        self.context_time = timing["context_time"]
        self.total_tokens = timing["total_tokens"]
        self.generation_time_list = timing["generation_time_list"]
        self.generation_tokens = len(self.generation_time_list)
        self.average_speed = (self.context_time + np.sum(self.generation_time_list)) / (
            self.context_tokens + self.generation_tokens
        )

    def show(self):
        if self.total_tokens == 0:
            # No stats to show.
            return

        print("*" * 50)
        print(
            f"Speed of Generation : {np.average(self.generation_time_list)*1000:.3f} ms/token"
        )
        print("*" * 50)


def stream_output(output_stream, time_stats: TimeStats = None):
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            print(" ".join(output_text[pre:now]), end=" ", flush=True)
            pre = now
    print(" ".join(output_text[pre:]), flush=True)
    if "timing" in outputs and outputs["timing"] is not None:
        timing = outputs["timing"]
        total_tokens = timing["total_tokens"]
        if time_stats is not None:
            time_stats.update(timing)
        prompt_tokens = timing["context_tokens"]
    print("-" * 50)
    print("TTFT: {:.3f} s for {} tokens.".format(timing["context_time"], prompt_tokens))
    return " ".join(output_text), total_tokens
