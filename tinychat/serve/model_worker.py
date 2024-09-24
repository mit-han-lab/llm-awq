#    Modified from https://github.com/haotian-liu/LLaVA
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial
from tqdm import tqdm

import tinychat.utils.constants
from tinychat.utils.constants import (
    WORKER_HEART_BEAT_INTERVAL,
    LLAVA_DEFAULT_IMAGE_TOKEN_IDX,
    LLAVA_DEFAULT_IMAGE_TOKEN,
    LLAVA_DEFAULT_IM_START_TOKEN,
    LLAVA_DEFAULT_IM_END_TOKEN,
)
from tinychat.utils.log_utils import (
    build_logger,
    server_error_msg,
    pretty_print_semaphore,
)
from tinychat.stream_generators.llava_stream_gen import tokenizer_image_token
from tinychat.utils.llava_image_processing import process_images, load_image_from_base64
from tinychat.models.llava_llama import LlavaLlamaForCausalLM
from tinychat.stream_generators.llava_stream_gen import LlavaStreamGenerator
from tinychat.utils.prompt_templates import (
    get_prompter,
    get_stop_token_ids,
    get_image_token,
)
from tinychat.utils.conversation_utils import gen_params

from transformers import AutoConfig, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def skip(*args, **kwargs):
    pass


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        model_type,
        model_path,
        model_name,
        quant_path,
        precision,
        device,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_type = model_type
        self.model_path = model_path
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith("checkpoint-"):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name
        if precision == "W4A16":
            self.model_name = self.model_name + "-4bit-AWQ"
        self.device = device

        # Load TinyChat model
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")

        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.kaiming_normal_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX = (
            self.tokenizer.convert_tokens_to_ids(
                [tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN]
            )[0]
        )
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.min_max_range_path = args.model_path + "/emb_min_max.pt"
        model = LlavaLlamaForCausalLM(config, args.device).half()
        vision_tower = model.get_model().vision_tower
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower = vision_tower.half()
        self.image_processor = vision_tower.image_processor

        if precision == "W16A16":
            pbar = tqdm(range(1))
            pbar.set_description("Loading checkpoint shards")
            for i in pbar:
                model = load_checkpoint_and_dispatch(
                    model,
                    model_path,
                    no_split_module_classes=[
                        "OPTDecoderLayer",
                        "LlamaDecoderLayer",
                        "BloomBlock",
                        "MPTBlock",
                        "DecoderLayer",
                        "CLIPEncoderLayer",
                    ],
                ).to(device)
        elif precision == "W4A16":
            from tinychat.utils.load_quant import load_awq_model

            model = load_awq_model(model, quant_path, 4, 128, device)
            from tinychat.modules import (
                make_quant_norm,
                make_quant_attn,
            )

            make_quant_attn(model, device)
            make_quant_norm(model)
            model = model.to(device)
        else:
            raise NotImplementedError(f"Precision {precision} is not supported.")

        self.model = model
        self.is_multimodal = (
            "llava" in self.model_name.lower() or "vila" in self.model_name.lower()
        )

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}"
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + (
                    len(model_semaphore._waiters)
                    if model_semaphore._waiters is not None
                    else 0
                )
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = (
            self.tokenizer,
            self.model,
            self.image_processor,
        )

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(LLAVA_DEFAULT_IMAGE_TOKEN):
                    raise ValueError(
                        "Number of images does not match number of <image> tokens in prompt"
                    )

                images = [load_image_from_base64(image) for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [
                        image.to(model.device, dtype=torch.float16) for image in images
                    ]
                else:
                    images = images.to(model.device, dtype=torch.float16)

                replace_token = LLAVA_DEFAULT_IMAGE_TOKEN
                if getattr(model.config, "mm_use_im_start_end", False):
                    replace_token = (
                        LLAVA_DEFAULT_IM_START_TOKEN
                        + replace_token
                        + LLAVA_DEFAULT_IM_END_TOKEN
                    )
                prompt = prompt.replace(LLAVA_DEFAULT_IMAGE_TOKEN, replace_token)
            else:
                images = None
        else:
            images = None

        gen_params.temp = float(params.get("temperature", 1.0))
        gen_params.top_p = float(params.get("top_p", 1.0))
        gen_params.n_predict = min(int(params.get("max_new_tokens", 256)), 1024)

        stream_generator = LlavaStreamGenerator
        stop_token_ids = get_stop_token_ids(self.model_type, self.model_path)
        image_token = get_image_token(model, self.model_path)
        image_token_holder = (
            tinychat.utils.constants.LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER
        )
        prompt = prompt.replace(image_token_holder, image_token)

        # print("=" * 50)
        # print(prompt)
        # print('=' * 50)
        output_stream = stream_generator(
            model,
            tokenizer,
            prompt,
            gen_params,
            device=model.device,
            stop_token_ids=stop_token_ids,
            image_tensor=images,
        )

        generated_text = ori_prompt
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                generated_text += " ".join(output_text[pre:now]) + " "
                yield json.dumps(
                    {"text": generated_text, "error_code": 0}
                ).encode() + b"\0"
                pre = now
        generated_text += " ".join(output_text[pre:])
        yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(
        partial(release_model_semaphore, fn=worker.send_heart_beat)
    )
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LLaMa",
        help="type of the (base) language model",
    )
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--quant-path", type=str, default=None)
    parser.add_argument("--precision", type=str, default="W4A16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--multi-modal",
        action="store_true",
        help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning(
            "Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path."
        )

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_type,
        args.model_path,
        args.model_name,
        args.quant_path,
        args.precision,
        args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
