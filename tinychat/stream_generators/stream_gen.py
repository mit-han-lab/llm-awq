import torch
import gc
import time

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

context_tokens = 0
context_time = 0.0
total_tokens = 0
generation_time_list = []


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def StreamGenerator(
    model,
    tokenizer,
    input: str,
    gen_params: dict,
    device: str = "cuda:0",
    stream_interval: int = 2,
    echo: bool = False,
    stop_token_ids=[],
):
    input_ids = tokenizer(input).input_ids
    input_echo_len = len(input_ids)
    # print(input_ids)
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
    start_pos = 0
    for i in range(max_new_tokens):
        torch.cuda.synchronize()
        t_st = time.time()

        if i == 0:
            inputs = torch.as_tensor([input_ids], device=device)
        else:
            inputs = torch.as_tensor([[token]], device=device)

        if (
            "llama" not in model.__class__.__name__.lower()
            and "mpt" not in model.__class__.__name__.lower()
            and "falcon" not in model.__class__.__name__.lower()
        ):
            if i == 0:  # Context Stage
                out = model(inputs, use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=inputs,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            out = model(inputs, start_pos=start_pos)
            start_pos += out.shape[1]
            logits = out
        torch.cuda.synchronize()
        t_ed = time.time()

        # Processing the logits
        if logits_processor:
            if gen_params.repeat_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]
        if gen_params.temp < 1e-5 or gen_params.top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
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
            context_tokens = logits.shape[1]
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
