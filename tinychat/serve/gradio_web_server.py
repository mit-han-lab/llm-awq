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

import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

LOGDIR = "."

from tinychat.serve.llava_conv import (
    default_conversation,
    conv_templates,
    get_conversation,
    SeparatorStyle,
)
from tinychat.utils.log_utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
import hashlib

IMAGE_BOX_NUM = 3
BUTTON_LIST_LEN = 2

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "TinyChat AWQ Chatbot"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

from tinychat.utils.constants import (
    LLAVA_DEFAULT_IMAGE_TOKEN,
    LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER,
    AUTO_FILL_IM_TOKEN_HOLDER,
)

# IMAGE_TOKEN_VIS = "**[IMAGE]**"
IMAGE_TOKEN_VIS = "**\<image\>**"

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, prompt_style_btn, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)
    state = get_conversation(prompt_style_btn)
    # state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(prompt_style_btn, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = get_conversation(prompt_style_btn)
    # state = default_conversation.copy()
    dropdown_update = gr.Dropdown.update(
        choices=models, value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


# def vote_last_response(state, vote_type, model_selector, request: gr.Request):
#     with open(get_conv_log_filename(), "a") as fout:
#         data = {
#             "tstamp": round(time.time(), 4),
#             "type": vote_type,
#             "model": model_selector,
#             "state": state.dict(),
#             "ip": request.client.host,
#         }
#         fout.write(json.dumps(data) + "\n")


# def upvote_last_response(state, model_selector, request: gr.Request):
#     logger.info(f"upvote. ip: {request.client.host}")
#     vote_last_response(state, "upvote", model_selector, request)
#     return ("",) + (disable_btn,) * 3


# def downvote_last_response(state, model_selector, request: gr.Request):
#     logger.info(f"downvote. ip: {request.client.host}")
#     vote_last_response(state, "downvote", model_selector, request)
#     return ("",) + (disable_btn,) * 3


# def flag_last_response(state, model_selector, request: gr.Request):
#     logger.info(f"flag. ip: {request.client.host}")
#     vote_last_response(state, "flag", model_selector, request)
#     return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * BUTTON_LIST_LEN


def change_prompt_style(state, prompt_style_btn, request: gr.Request):
    if state.version != prompt_style_btn:
        state = get_conversation(prompt_style_btn)
    return state


def clear_history(prompt_style_btn, request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = get_conversation(prompt_style_btn)
    return (
        (state, state.to_gradio_chatbot(), "")
        + (None,) * IMAGE_BOX_NUM
        + (None,)  # Videobox
        + (disable_btn,) * BUTTON_LIST_LEN
    )


def clear_text_history(state, prompt_style_btn, request: gr.Request):
    state = get_conversation(prompt_style_btn)
    return (state, state.to_gradio_chatbot())


def clear_after_click_example_1_video(videobox, textbox):
    imagebox = None
    imagebox_2 = None
    imagebox_3 = None
    state = get_conversation("default")
    prompt_style_btn = "default"
    return (state, imagebox, imagebox_2, imagebox_3, videobox, prompt_style_btn)


def clear_after_click_example_1_image(imagebox, textbox):
    imagebox_2 = None
    imagebox_3 = None
    videobox = None
    state = get_conversation("default")
    prompt_style_btn = "default"
    return (state, imagebox, imagebox_2, imagebox_3, videobox, prompt_style_btn)


def clear_after_click_example_2_image(imagebox, imagebox_2, textbox):
    imagebox_3 = None
    videobox = None
    state = get_conversation("default")
    prompt_style_btn = "default"
    return (state, imagebox, imagebox_2, imagebox_3, videobox, prompt_style_btn)


def clear_after_click_example_3_image(imagebox, imagebox_2, imagebox_3, textbox):
    videobox = None
    state = get_conversation("default")
    prompt_style_btn = "default"
    return (state, imagebox, imagebox_2, imagebox_3, videobox, prompt_style_btn)


def clear_after_click_example_3_image_icl(imagebox, imagebox_2, imagebox_3, textbox):
    videobox = None
    state = get_conversation("no-sys")
    prompt_style_btn = "no-sys"
    return (state, imagebox, imagebox_2, imagebox_3, videobox, prompt_style_btn)


def add_images(
    state,
    imagebox,
    imagebox_2,
    imagebox_3,
    videobox,
    image_process_mode,
    request: gr.Request,
):
    if state.image_loaded:
        # return (state,) + (None,) * IMAGE_BOX_NUM
        return state

    def extract_frames(video_path):
        import cv2
        from PIL import Image

        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        frame_interval = frame_count // 8
        print(
            "duration:", duration, "frames:", frame_count, "intervals:", frame_interval
        )
        # frame_interval = 10

        def get_frame(max_frames):
            # frame_id = int(fps * stamp)
            # vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            # ret, frame = vidcap.read()
            images = []
            count = 0
            success = True
            while success:
                success, frame = vidcap.read()
                if count % frame_interval == 0:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                    if len(images) == max_frames:
                        return images

                count += 1
            # assert ret, "videocap.read fails!"
            # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # im_pil = Image.fromarray(img)
            # print(f"loading {stamp} success")
            return images

        # return [get_frame(0), get_frame(stamp1), get_frame(stamp2)]
        # img = get_frame(0)
        # img1 = get_frame(frame_interval * 1)
        # return [img, img1, img, img1, img, img1,]
        return get_frame(8)

    frames = [
        None,
    ]
    if videobox is not None:
        frames = extract_frames(videobox)
        # add frames as regular images
        logger.info(f"Got videobox: {videobox}.")

    logger.info(f"add_image. ip: {request.client.host}.")
    image_list = [imagebox, imagebox_2, imagebox_3, *frames]
    logger.info(f"image_list: {image_list}")

    im_count = 0
    for image in image_list:
        if image is not None:
            im_count += 1
    for image in image_list:
        if image is not None:
            if args.auto_pad_image_token or im_count == 1:
                text = (AUTO_FILL_IM_TOKEN_HOLDER, image, image_process_mode)
            else:
                text = ("", image, image_process_mode)
            state.append_message(None, text)
            state.append_message(
                None, None
            )  # in order to match the input-output pair for textbox outputs
            # state.append_message(state.roles[0], text)
            # state.append_message(state.roles[1], None)
    # state.skip_next = False
    logger.info(f"im_count {im_count}. ip: {request.client.host}.")
    state.image_loaded = True
    # return (state,) + (None,) * IMAGE_BOX_NUM
    return state


def add_text_only(state, text, request: gr.Request):
    logger.info(f"add_text_only. ip: {request.client.host}. len: {len(text)}")

    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, moderation_msg) + (no_change_btn,) * BUTTON_LIST_LEN

    # This is 1536 characters, rather than tokens
    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, "") + (disable_btn,) * BUTTON_LIST_LEN


def add_text(
    state, text, image, image_process_mode, prompt_style_btn, request: gr.Request
):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (
            no_change_btn,
        ) * BUTTON_LIST_LEN
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,
            ) * BUTTON_LIST_LEN

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            # text = '<Image><image></Image>' + text
            text = text + "\n<image>"
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = get_conversation(prompt_style_btn)
            # state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (
        disable_btn,
    ) * BUTTON_LIST_LEN


def http_bot(
    state,
    model_selector,
    temperature,
    top_p,
    max_new_tokens,
    prompt_style_btn,
    request: gr.Request,
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * BUTTON_LIST_LEN
        return
    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if "llama-2" in model_name.lower():
                template_name = "llava_llama_2"
            elif "v1" in model_name.lower():
                if "mmtag" in model_name.lower():
                    template_name = "v1_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if "mmtag" in model_name.lower():
                    template_name = "v0_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        if prompt_style_btn == "no-sys":
            new_state = get_conversation(prompt_style_btn)
        else:
            new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            # disable_btn,
            # disable_btn,
            # disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg"
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": (
            state.sep
            if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
            else state.sep2
        ),
        "images": f"List of {len(state.get_images())} images: {all_image_hash}",
    }

    image_num = len(state.get_images())
    if image_num == 0:
        state.messages[-1][
            -1
        ] = "**NO INPUT IMAGE RECEIVED BY THE SERVER. PLEASE CHECK YOUR INTERNET CONNECTION AND REFRESH THE PAGE.**"
        yield (
            state,
            state.to_gradio_chatbot(),
            # disable_btn,
            # disable_btn,
            # disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    count_auto_im_token = prompt.count(AUTO_FILL_IM_TOKEN_HOLDER)
    count_manual_im_token = prompt.count(LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER)
    if (count_auto_im_token == image_num) and (
        count_manual_im_token == 0
    ):  # Use default system prompt
        prompt = prompt.replace(AUTO_FILL_IM_TOKEN_HOLDER, LLAVA_DEFAULT_IMAGE_TOKEN)
    elif (count_auto_im_token == image_num) and (
        count_manual_im_token == image_num
    ):  # Use <image> token inserted by user
        prompt = prompt.replace(AUTO_FILL_IM_TOKEN_HOLDER, "")
        prompt = prompt.replace(
            LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER, LLAVA_DEFAULT_IMAGE_TOKEN
        )
    elif (count_auto_im_token == 0) and (count_manual_im_token == image_num):
        prompt = prompt.replace(
            LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER, LLAVA_DEFAULT_IMAGE_TOKEN
        )
    else:
        state.messages[-1][
            -1
        ] = "**IMAGE NUM MISMATCHES IMAGE TOKEN PLACEHOLDER. PLEASE CHECK YOUR INPUT AND REFRESH THE PAGE.**"
        yield (
            state,
            state.to_gradio_chatbot(),
            # disable_btn,
            # disable_btn,
            # disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    pload["prompt"] = prompt
    logger.info(f"==== request ====\n{pload}")
    pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    ret = state.to_gradio_chatbot()
    ret[0][0] = ret[0][0].replace(LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER, IMAGE_TOKEN_VIS)
    yield (state, ret) + (disable_btn,) * BUTTON_LIST_LEN

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=10,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) :].strip()
                    state.messages[-1][-1] = output + "▌"
                    ret = state.to_gradio_chatbot()
                    ret[0][0] = ret[0][0].replace(
                        LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER, IMAGE_TOKEN_VIS
                    )
                    yield (state, ret) + (disable_btn,) * BUTTON_LIST_LEN
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    ret = state.to_gradio_chatbot()
                    ret[0][0] = ret[0][0].replace(
                        LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER, IMAGE_TOKEN_VIS
                    )
                    yield (state, ret) + (
                        # disable_btn,
                        # disable_btn,
                        # disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            # disable_btn,
            # disable_btn,
            # disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    ret = state.to_gradio_chatbot()
    ret[0][0] = ret[0][0].replace(LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER, IMAGE_TOKEN_VIS)
    yield (state, ret) + (enable_btn,) * BUTTON_LIST_LEN

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


title_markdown = """
# VILA: On Pre-training for Visual Language Models
[\[Paper\]](https://arxiv.org/abs/2312.07533)  [\[Github\]](https://github.com/NVlabs/VILA)
### Powered by [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) with 4-bit [AWQ](https://arxiv.org/abs/2306.00978).
"""

tos_markdown = """
### Terms of Use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""


learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""

ack_markdown = """
### Acknowledgement
This demo is inspired by [LLaVA](https://github.com/haotian-liu/LLaVA). We thank LLaVA for providing an elegant way to build the Gradio Web UI.          
"""

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def build_demo(embed_mode):
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    with gr.Blocks(
        title="VILA on TinyChat", theme=gr.themes.Default(), css=block_css
    ) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=8):
                with gr.Row():
                    imagebox = gr.Image(type="pil")
                    imagebox_2 = gr.Image(type="pil")
                    imagebox_3 = gr.Image(type="pil")
                    videobox = gr.Video(label="1 video = 8 frames")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False,
                )
                # imagebox_out = gr.Image(height=150)
                with gr.Row():
                    with gr.Column(scale=5):
                        textbox.render()
                    with gr.Column(scale=1, min_width=100):
                        submit_btn = gr.Button(value="Send", variant="primary")
                    with gr.Column(scale=1, min_width=100):
                        clear_btn = gr.Button(
                            value="🗑️  Clear", variant="primary", interactive=False
                        )
                    with gr.Column(scale=1, min_width=100):
                        regenerate_btn = gr.Button(
                            value="🔄  Retry", variant="primary", interactive=False
                        )
                with gr.Row():
                    gr.Markdown(
                        "### *** Before changing the current images, uploading new images or switching the prompt style, please click the clear button."
                    )
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="TinyChat Assistant", height=550
                )

            with gr.Column(scale=4):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=50):
                        model_selector = gr.Dropdown(
                            choices=models,
                            value=models[0] if len(models) > 0 else "",
                            label="Model",
                            interactive=True,
                            show_label=True,
                            container=False,
                        )
                    with gr.Column(scale=1, min_width=50):
                        prompt_style_btn = gr.Radio(
                            ["default", "no-sys"],
                            label="Prompt style",
                            value="default",
                            interactive=True,
                        )

                # with gr.Row():
                # with gr.Column(scale=1, min_width=50):
                #     im_submit_btn = gr.Button(value="Add image", variant="primary")
                # with gr.Column(scale=1, min_width=50):
                #     submit_btn_1 = gr.Button(value="Send", variant="primary")

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                with gr.Row(equal_height=True):
                    gr.Examples(
                        examples=[
                            [
                                f"{cur_dir}/examples/video/qZDF__7LNKc.4.mp4",
                                "Elaborate on the visual and narrative elements of the video in detail.",
                            ],
                        ],
                        label="Video Example",
                        inputs=[videobox, textbox],
                        fn=clear_after_click_example_1_video,
                        outputs=[
                            state,
                            imagebox,
                            imagebox_2,
                            imagebox_3,
                            videobox,
                            prompt_style_btn,
                        ],
                        run_on_click=True,
                    )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=50):
                        gr.Examples(
                            examples=[
                                [
                                    f"{cur_dir}/examples/pedestrain.png",
                                    "<image> What is the person in the center of the image doing?",
                                ],
                            ],
                            label="Image Example 1",
                            inputs=[imagebox, textbox],
                            fn=clear_after_click_example_1_image,
                            outputs=[
                                state,
                                imagebox,
                                imagebox_2,
                                imagebox_3,
                                videobox,
                                prompt_style_btn,
                            ],
                            run_on_click=True,
                        )
                    with gr.Column(scale=1, min_width=50):
                        gr.Examples(
                            examples=[
                                [
                                    f"{cur_dir}/examples/car_repair.png",
                                    "<image> What is the brand of the silver car in the image?",
                                ],
                            ],
                            label="Image Example 2",
                            inputs=[imagebox, textbox],
                            fn=clear_after_click_example_1_image,
                            outputs=[
                                state,
                                imagebox,
                                imagebox_2,
                                imagebox_3,
                                videobox,
                                prompt_style_btn,
                            ],
                            run_on_click=True,
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=50):
                        gr.Examples(
                            examples=[
                                [
                                    f"{cur_dir}/examples/CPR.jpg",
                                    "<image> What are the people doing in this image?",
                                ],
                            ],
                            label="Image Example 3",
                            inputs=[imagebox, textbox],
                            fn=clear_after_click_example_1_image,
                            outputs=[
                                state,
                                imagebox,
                                imagebox_2,
                                imagebox_3,
                                videobox,
                                prompt_style_btn,
                            ],
                            run_on_click=True,
                        )
                    with gr.Column(scale=1, min_width=50):
                        gr.Examples(
                            examples=[
                                [
                                    f"{cur_dir}/examples/Wall_fissure.png",
                                    "<image> What are the likely service needed for this building?",
                                ],
                            ],
                            label="Image Example 4",
                            inputs=[imagebox, textbox],
                            fn=clear_after_click_example_1_image,
                            outputs=[
                                state,
                                imagebox,
                                imagebox_2,
                                imagebox_3,
                                videobox,
                                prompt_style_btn,
                            ],
                            run_on_click=True,
                        )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=50):
                        gr.Examples(
                            examples=[
                                [
                                    f"{cur_dir}/examples/animal_blocking.png",
                                    "<image> What is unusual in this image?",
                                ],
                            ],
                            label="Image Example 5",
                            inputs=[imagebox, textbox],
                            fn=clear_after_click_example_1_image,
                            outputs=[
                                state,
                                imagebox,
                                imagebox_2,
                                imagebox_3,
                                videobox,
                                prompt_style_btn,
                            ],
                            run_on_click=True,
                        )
                    with gr.Column(scale=1, min_width=50):
                        gr.Examples(
                            examples=[
                                [
                                    f"{cur_dir}/examples/windmill_people.png",
                                    "<image> Can you describe what is happening?",
                                ],
                            ],
                            label="Image Example 6",
                            inputs=[imagebox, textbox],
                            fn=clear_after_click_example_1_image,
                            outputs=[
                                state,
                                imagebox,
                                imagebox_2,
                                imagebox_3,
                                videobox,
                                prompt_style_btn,
                            ],
                            run_on_click=True,
                        )

                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/climate_change/climate_change_1.png",
                            f"{cur_dir}/examples/climate_change/climate_change_2.png",
                            "<image> <image> What is the implication of temperature based on this image?",
                        ],
                    ],
                    inputs=[imagebox, imagebox_2, textbox],
                    label="Multi-image Example 1",
                    fn=clear_after_click_example_2_image,
                    outputs=[
                        state,
                        imagebox,
                        imagebox_2,
                        imagebox_3,
                        videobox,
                        prompt_style_btn,
                    ],
                    run_on_click=True,
                )

                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/palms/palm1.png",
                            f"{cur_dir}/examples/palms/palm2.png",
                            f"{cur_dir}/examples/palms/palm3.png",
                            "8:15am: <image> 12:45pm: <image> 16:00pm: <image> When did I have lunch and what did I eat for lunch?",
                        ],
                    ],
                    inputs=[imagebox, imagebox_2, imagebox_3, textbox],
                    label="Multi-image Example 2",
                    fn=clear_after_click_example_3_image,
                    outputs=[
                        state,
                        imagebox,
                        imagebox_2,
                        imagebox_3,
                        videobox,
                        prompt_style_btn,
                    ],
                    run_on_click=True,
                )

                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/golf/Golfman1.png",
                            f"{cur_dir}/examples/golf/Golfman2.png",
                            f"{cur_dir}/examples/golf/Golfman3.png",
                            "<image> <image> <image> What happens to the man after hitting the ball? And why does it happen?",
                        ],
                    ],
                    inputs=[imagebox, imagebox_2, imagebox_3, textbox],
                    label="Multi-image Example 3",
                    fn=clear_after_click_example_3_image,
                    outputs=[
                        state,
                        imagebox,
                        imagebox_2,
                        imagebox_3,
                        videobox,
                        prompt_style_btn,
                    ],
                    run_on_click=True,
                )

                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/icl-logo/google.webp",
                            f"{cur_dir}/examples/icl-logo/apple.jpg",
                            f"{cur_dir}/examples/icl-logo/nvidia.png",
                            "<image> is famous for its search engine. <image> is famous for Mac and iPhone. <image> ",
                        ],
                    ],
                    inputs=[imagebox, imagebox_2, imagebox_3, textbox],
                    label="In-context Learning Example 1 (Please switch the prompt style to 'no-sys')",
                    fn=clear_after_click_example_3_image_icl,
                    outputs=[
                        state,
                        imagebox,
                        imagebox_2,
                        imagebox_3,
                        videobox,
                        prompt_style_btn,
                    ],
                    run_on_click=True,
                )

                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/icl-building/csail_building.jpeg",
                            f"{cur_dir}/examples/icl-building/Toronto_Tower.jpeg",
                            f"{cur_dir}/examples/icl-building/Golden_State_Bridge.jpeg",
                            "<image> Boston. <image> Toronto. <image> ",
                        ],
                    ],
                    inputs=[imagebox, imagebox_2, imagebox_3, textbox],
                    label="In-context Learning Example 2 (Please switch the prompt style to 'no-sys')",
                    fn=clear_after_click_example_3_image_icl,
                    outputs=[
                        state,
                        imagebox,
                        imagebox_2,
                        imagebox_3,
                        videobox,
                        prompt_style_btn,
                    ],
                    run_on_click=True,
                )

                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/arts/sunflowers.jpg",
                            f"{cur_dir}/examples/arts/the_persistence_of_memory.png",
                            f"{cur_dir}/examples/arts/impression_sunrise.png",
                            "<image> Vincent Van Gogh. <image> Salvador Dalí. <image>",
                        ],
                    ],
                    inputs=[imagebox, imagebox_2, imagebox_3, textbox],
                    label="In-context Learning Example 3 (Please switch the prompt style to 'no-sys')",
                    fn=clear_after_click_example_3_image_icl,
                    outputs=[
                        state,
                        imagebox,
                        imagebox_2,
                        imagebox_3,
                        videobox,
                        prompt_style_btn,
                    ],
                    run_on_click=True,
                )

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

                # with gr.Row(elem_id="buttons") as button_row:
                # upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                # downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                # flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                # regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
            gr.Markdown(ack_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        # btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        btn_list = [regenerate_btn, clear_btn]

        # upvote_btn.click(
        #     upvote_last_response,
        #     [state, model_selector],
        #     [textbox, upvote_btn, downvote_btn, flag_btn],
        #     queue=False,
        # )
        # downvote_btn.click(
        #     downvote_last_response,
        #     [state, model_selector],
        #     [textbox, upvote_btn, downvote_btn, flag_btn],
        #     queue=False,
        # )
        # flag_btn.click(
        #     flag_last_response,
        #     [state, model_selector],
        #     [textbox, upvote_btn, downvote_btn, flag_btn],
        #     queue=False,
        # )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox] + btn_list,
            queue=False,
        ).then(
            http_bot,
            [
                state,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,
                prompt_style_btn,
            ],
            [state, chatbot] + btn_list,
        )

        prompt_style_btn.change(
            change_prompt_style, [state, prompt_style_btn], [state], queue=False
        )

        clear_btn.click(
            clear_history,
            [prompt_style_btn],
            [state, chatbot, textbox, imagebox, imagebox_2, imagebox_3, videobox]
            + btn_list,
            queue=False,
        )

        # textbox.submit(
        #     add_text,
        #     [state, textbox, imagebox, image_process_mode],
        #     [state, chatbot, textbox, imagebox] + btn_list,
        #     queue=False
        # ).then(
        #     http_bot,
        #     [state, model_selector, temperature, top_p, max_output_tokens],
        #     [state, chatbot] + btn_list
        # )

        # im_submit_btn.click(
        #     mirror,
        #     inputs=[imagebox],
        #     outputs=[imagebox_out]
        # ).then(
        #     add_image,
        #     [state, imagebox, image_process_mode],
        #     [state, imagebox] + btn_list,
        #     queue=False
        # )

        textbox.submit(
            clear_text_history, [state, prompt_style_btn], [state, chatbot], queue=False
        ).then(
            add_images,
            [state, imagebox, imagebox_2, imagebox_3, videobox, image_process_mode],
            [state],
            queue=False,
        ).then(
            add_text_only, [state, textbox], [state, textbox] + btn_list, queue=False
        ).then(
            http_bot,
            [
                state,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,
                prompt_style_btn,
            ],
            [state, chatbot] + btn_list,
        )

        submit_btn.click(
            clear_text_history, [state, prompt_style_btn], [state, chatbot], queue=False
        ).then(
            add_images,
            [state, imagebox, imagebox_2, imagebox_3, videobox, image_process_mode],
            [state],
            queue=False,
        ).then(
            add_text_only, [state, textbox], [state, textbox] + btn_list, queue=False
        ).then(
            http_bot,
            [
                state,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,
                prompt_style_btn,
            ],
            [state, chatbot] + btn_list,
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params, prompt_style_btn],
                [state, model_selector],
                _js=get_window_url_params,
                queue=False,
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                [prompt_style_btn],
                [state, model_selector],
                queue=False,
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument(
        "--auto-pad-image-token",
        action="store_true",
        help="Automatically pad <image> token to the before of the prompt if no user inputs.",
    )
    # NOTE: For single image input, we still auto pad <image> token even if the --auto-pad-image-token is False
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed)
    demo.queue(concurrency_count=args.concurrency_count, api_open=False).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )
