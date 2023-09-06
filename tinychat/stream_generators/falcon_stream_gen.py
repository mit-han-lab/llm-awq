import gc
from threading import Thread
from typing import Iterable

import torch
import transformers
from transformers import TextIteratorStreamer, GenerationConfig


transformers.logging.set_verbosity_error()


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


@torch.inference_mode()
def FalconStreamGenerator(
    model,
    tokenizer,
    input: str,
    gen_params: dict,
    device: str = "cuda:0",
    context_len=2048,
    stream_interval=2,
    judge_sent_end=False,
    echo: bool = False,
    stop_str: str = "\nUser",
    stop_token_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    prompt = input
    len_prompt = len(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if gen_params.top_k <= 0:
        top_k = gen_params.n_vocab
    else:
        top_k = gen_params.top_k
    max_new_tokens = gen_params.n_predict
    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]  # truncate the input prompt
    attention_mask = attention_mask[-max_src_len:]  # truncate the input prompt
    input_echo_len = len(input_ids)
    stop_token_ids.append(tokenizer.eos_token_id)

    decode_config = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, **decode_config)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=gen_params.temp >= 1e-5,
        temperature=gen_params.temp,
        repetition_penalty=gen_params.repeat_penalty,
        no_repeat_ngram_size=10,
        top_p=gen_params.top_p,
        top_k=top_k,
        eos_token_id=stop_token_ids,
    )

    generation_kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        generation_config=generation_config,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    if echo:
        # means keep the prompt
        output = prompt
    else:
        output = ""

    for i, new_text in enumerate(streamer):
        output += new_text
        if i % stream_interval == 0:
            if echo:
                rfind_start = len_prompt
            else:
                rfind_start = 0

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }
    output = output.strip()

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif partially_stopped:
        finish_reason = None
    else:
        finish_reason = "stop"

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    gc.collect()
    torch.cuda.empty_cache()
