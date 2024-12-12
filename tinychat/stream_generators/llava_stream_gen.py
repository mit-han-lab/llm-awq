import torch
import gc
import time
from typing import Optional

import tinychat.utils.constants

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
# )

context_tokens = 0
context_time = 0.0
total_tokens = 0
generation_time_list = []


def prepare_logits_processor(
    temperature: float,
    repetition_penalty: float,
    top_p: float,
    top_k: int,
    min_tokens_to_keep: int = 1,
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    # Removed for the newest version of VILA
    # if repetition_penalty > 1.0:
    #     processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(
            TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep)
        )
    return processor_list


#  This function is inspired by https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py#L185
def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_TOKEN_IDX,
    return_tensors=None,
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


@torch.inference_mode()
def LlavaStreamGenerator(
    model,
    tokenizer,
    input: str,
    start_pos: int,
    gen_params: dict,
    device: str = "cuda:0",
    stream_interval: int = 1,
    echo: bool = False,
    stop_token_ids=[],
    image_tensor: Optional[torch.FloatTensor] = None,
    chunk_prefilling: bool = False,
):
    if chunk_prefilling and start_pos != 0:  # </s>USER:2,11889 while USER:3148,1001
        input = "</s>" + input
    input_ids = (
        tokenizer_image_token(
            input,
            tokenizer,
            tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_TOKEN_IDX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(device)
    )
    if chunk_prefilling and start_pos != 0:
        input_ids = input_ids[
            :, 2:
        ]  # tokenizer will add a <s> at the beginning, so to delete it
    special_token = "<image>" in input
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    len_input = len(input)
    if gen_params.top_k <= 0:
        top_k = gen_params.n_vocab
    else:
        top_k = gen_params.top_k
    logits_processor = prepare_logits_processor(
        gen_params.temp, gen_params.repeat_penalty, gen_params.top_p, top_k
    )

    past_key_values = out = None
    stop_token_ids.append(tokenizer.eos_token_id)
    max_new_tokens = gen_params.n_predict

    batch_size = 1  # TODO: support multi-batch
    position_ids = [
        torch.arange(
            start_pos, start_pos + input_ids.numel(), dtype=torch.long, device=device
        )
        for i in range(batch_size)
    ]
    position_ids = torch.stack(position_ids)

    for i in range(max_new_tokens):
        torch.cuda.synchronize()
        t_st = time.time()

        if i == 0:
            # inputs = torch.as_tensor([input_ids], device=device)
            inputs = input_ids
        else:
            position_ids = (position_ids[:, -1] + 1).reshape(
                1, 1
            )  # [Important] fixed the bug of positions
            inputs = torch.as_tensor([[token]], device=device)

        attention_mask = torch.ones(size=inputs.shape, dtype=torch.int, device=device)

        if (
            "llama" not in model.__class__.__name__.lower()
            and "mpt" not in model.__class__.__name__.lower()
            and "falcon" not in model.__class__.__name__.lower()
            and "llava" not in model.__class__.__name__.lower()
            # and "vila" not in model.__class__.__name__.lower()    # VILA model reuses the model class of LLaVA
        ):
            if i == 0:  # Context Stage
                out = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    images=image_tensor,
                    return_dict=True,
                    # special_token=special_token,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_attentions=False,
                    output_hidden_states=False,
                    images=image_tensor,
                    return_dict=True,
                    # special_token=special_token,
                )
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            out = model(
                input_ids=inputs,
                start_pos=start_pos,
                images=image_tensor,
                position_ids=position_ids,
                attention_mask=attention_mask,
                special_token=special_token,
                chunk_prefilling=chunk_prefilling,
            )
            start_pos += (
                inputs.shape[1] + 195 * torch.sum(inputs[0] == IMAGE_TOKEN_INDEX).item()
            )
            logits = out
        torch.cuda.synchronize()
        t_ed = time.time()

        # Processing the logits
        if logits_processor:
            if gen_params.repeat_penalty > 1.0:
                # tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                tmp_output_ids = output_ids[0].unsqueeze(0)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[:, -1, :]
        if gen_params.temp < 1e-5 or gen_params.top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
            print(token)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
        output_ids.append(token)

        global context_time
        global context_tokens
        global total_tokens
        global generation_time_list
        if i == 0:
            context_time = t_ed - t_st
            context_tokens = (
                inputs.shape[1] + 195 * torch.sum(inputs[0] == IMAGE_TOKEN_INDEX).item()
            )
            generation_time_list = []
        else:
            generation_time_list.append(t_ed - t_st)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_input
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            partially_stopped = False

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
                    "timing": None,
                }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    total_tokens = context_tokens + len(generation_time_list)
    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
        "timing": {
            "context_tokens": context_tokens,
            "context_time": context_time,
            "total_tokens": total_tokens,
            "generation_time_list": generation_time_list,
        },
    }

    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()

    # return context_tokens, context_time, total_tokens, generation_time_list
