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

import torch
from PIL import Image
from io import BytesIO
import requests
import os
import base64


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def vis_images(image_files):
    if len(image_files) == 1:
        image = image_files[0]
        os.system(f"termvisage --query-timeout 1 {image} -H left --height 12")

    else:
        # Concat images
        system_inst = "convert "
        inst_template1 = " \\( {image} -background none -resize x500 \\) "
        inst_template2 = " \\( {image} -background none -resize x500 -splice 100x0 \\) "
        count = 0
        for image in image_files:
            count += 1
            if count == 1:
                system_inst += inst_template1.format(image=image)
            else:
                system_inst += inst_template2.format(image=image)
        system_inst += " +append .vis.jpg"
        os.system(system_inst)

        os.system(f"termvisage --query-timeout 1 .vis.jpg -H left")


def expand2square(pil_img, background_color):
    """
    Copy from Llava codebase for image preprocessing.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    """
    Copy from Llava codebase for image preprocessing.
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            if "intern" in image_processor.__class__.__name__.lower():
                # special case
                new_images.append(image.unsqueeze(0))
            else:
                new_images.append(image)
    else:
        ret = image_processor(images, return_tensors="pt")["pixel_values"]
        if "intern" in image_processor.__class__.__name__.lower():
            # special case
            ret = [x.unsqueeze(0) for x in ret]
        return ret
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)

    return new_images
