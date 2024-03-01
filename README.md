# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration 
[[Paper](https://arxiv.org/abs/2306.00978)][[Slides](https://www.dropbox.com/scl/fi/dtnp6h6y1mnp7g036axu6/AWQ-slide.pdf?rlkey=ffgh50hxhx8dmsnjiu8kef0ou&dl=0)][[Video](https://youtu.be/3dYLj9vjfA0)]

**Efficient and accurate** low-bit weight quantization (INT3/4) for LLMs, supporting **instruction-tuned** models and **multi-modal** LMs.

![overview](figures/overview.png)

The current release supports: 

- AWQ search for accurate quantization. 
- Pre-computed AWQ model zoo for LLMs (LLaMA, Llama2, OPT, CodeLlama, StarCoder, Vicuna, VILA, LLaVA; load to generate quantized weights).
- Memory-efficient 4-bit Linear in PyTorch.
- Efficient CUDA kernel implementation for fast inference (support context and decoding stage).
- Examples on 4-bit inference of an instruction-tuned model (Vicuna) and **multi-modal LM** (VILA).

**Thanks to AWQ, TinyChat can deliver more efficient responses with LLM/VLM chatbots through 4-bit inference.**

* TinyChat on RTX 4090 (3.4x faster than FP16):

![TinyChat on RTX 4090: W4A16 is 3.4x faster than FP16](./tinychat/figures/4090_example.gif)

* TinyChat on Jetson Orin (3.2x faster than FP16):
  
![TinyChat on Orin: W4A16 is 3.2x faster than FP16](./tinychat/figures/orin_example.gif)

**TinyChat also supports inference with vision language models (e.g., VILA, LLaVA). In the following examples, W4A16 quantized models from VILA family are launched with TinyChat.**

* TinyChat with VILA-13B on RTX 4090 (multi-image inputs supported):

![TinyChat with VILA on 4090](./tinychat/figures/4090_vila_example.gif)

* TinyChat with VILA-7B/13B on Jetson Orin:

![TinyChat with VILA on Orin](./tinychat/figures/orin_vila_example.gif)

<!-- Check out [TinyChat](tinychat), which delievers **30 tokens/second** inference performance (**3.2x faster** than FP16) for the **Llama2** chatbot on the resource-constrained NVIDIA Jetson Orin!  -->

Check out [TinyChat](tinychat), which offers a turn-key solution for **on-device inference** of LLMs and VLMs on **resource-constrained edge platforms**. With TinyChat, it is now possible to efficiently run **large** models on **small** and **low-power** devices even without Internet connection!


## News
- [2024/02] 🔥 AWQ has been accepted to **MLSys 2024**!
- [2024/02] 🔥 We supported [VILA Vision Languague Models](https://arxiv.org/abs/2312.07533) in AWQ & TinyChat! Check our latest demos with multi-image inputs!
- [2024/02] 🔥 We released new version of quantized GEMM/GEMV kernels in [**TinyChat**](tinychat), leading to **38 tokens/second** inference speed on NVIDIA Jetson Orin!
- [2023/11] 🔥 We added AWQ support and pre-computed search results for CodeLlama, StarCoder, StableCode models. Checkout our model zoo [here](https://huggingface.co/datasets/mit-han-lab/awq-model-zoo)!
- [2023/11] 🔥 AWQ is now integrated natively in Hugging Face transformers through `from_pretrained`. You can either load quantized models from the Hub or your own HF quantized models.
- [2023/10] AWQ is integrated into NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/)
- [2023/09] AWQ is integrated into [FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/awq.md), [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/quantization_utils/awq.py), [HuggingFace TGI](https://github.com/huggingface/text-generation-inference/pull/1054), and [LMDeploy](https://github.com/InternLM/lmdeploy). 
- [2023/09] ⚡ Check out our latest [**TinyChat**](tinychat), which is ~2x faster than the first release on Orin!
- [2023/09] ⚡ Check out [**AutoAWQ**](https://github.com/casper-hansen/AutoAWQ), a third-party implementation to make AWQ easier to expand to new models, improve inference speed, and integrate into Huggingface.
- [2023/07] 🔥 We released **TinyChat**, an efficient and lightweight chatbot interface based on AWQ. TinyChat enables efficient LLM inference on both cloud and edge GPUs. Llama-2-chat models are supported! Check out our implementation [here](tinychat).
- [2023/07] 🔥 We added AWQ support and pre-computed search results for Llama-2 models (7B & 13B). Checkout our model zoo [here](https://huggingface.co/datasets/mit-han-lab/awq-model-zoo)!
- [2023/07] We extended the support for more LLM models including MPT, Falcon, and BLOOM. 

## Contents

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](#awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration)
  - [News](#news)
  - [Contents](#contents)
  - [Install](#install)
  - [AWQ Model Zoo](#awq-model-zoo)
  - [Examples](#examples)
  - [Usage](#usage)
  - [Results on Vision-Language Models (VILA-7b/13B)](#results-on-vision-language-models-vila-7b13b)
  - [Inference speed ( Token/sec )](#inference-speed--tokensec-)
  - [Reference](#reference)
  - [Related Projects](#related-projects)

## Install

1. Clone this repository and navigate to AWQ folder
```
git clone https://github.com/mit-han-lab/llm-awq
cd llm-awq
```

2. Install Package
```
conda create -n awq python=3.10 -y
conda activate awq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

* For **edge devices** like Orin, before running the commands above, please:

    1. Modify [pyproject.toml](pyproject.toml) by commenting out [this line](https://github.com/mit-han-lab/llm-awq/blob/3fce69061682fdd528824e5da3d03a8a8b545f2a/pyproject.toml#L17).
    2. Set [this line](https://github.com/mit-han-lab/llm-awq/blob/3fce69061682fdd528824e5da3d03a8a8b545f2a/pyproject.toml#18) to transformers==4.32.0.
    3. Manually install precompiled PyTorch binaries (>=2.0.0) from [NVIDIA](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).
    4. Set the appropriate Python version for conda environment (e.g., `conda create -n awq python=3.8 -y` for JetPack 5).
  
3. Install efficient W4A16 (4-bit weight, 16-bit activation) CUDA kernel and optimized FP16 kernels (e.g. layernorm, positional encodings).
```
cd awq/kernels
python setup.py install
```

## AWQ Model Zoo

We provide pre-computed AWQ search results for multiple model families, including LLaMA, OPT, Vicuna, and LLaVA. To get the pre-computed AWQ search results, run:

```bash
# git lfs install  # install git lfs if not already
git clone https://huggingface.co/datasets/mit-han-lab/awq-model-zoo awq_cache
```

The detailed support list:

| Models | Sizes                       | INT4-g128 | INT3-g128 |
| ------ | --------------------------- | --------- | --------- |
| [Llama2](/scripts/llama_example.sh)  | 7B/13B/70B  | ✅         | ✅        |
| [LLaMA](/scripts/llama2_example.sh)  | 7B/13B/30B/65B              | ✅         | ✅        |
| [OPT](/scripts/opt_example.sh)    | 125m/1.3B/2.7B/6.7B/13B/30B | ✅         | ✅        |
| [CodeLlama](/scripts/codellama_example.sh) | 7B/13B/34B               | ✅         | ✅        |
| [StarCoder](/scripts/starcoder_example.sh) | 15.5B                    | ✅         | ✅        |
| [Vicuna-v1.1](/scripts/vicuna_example.sh) | 7B/13B                 | ✅         |           |
| [LLaVA-v0](/scripts/llava_example.sh) | 13B                       | ✅         |           |
| [VILA](/scripts/vila_example.sh)    | 7B/13B                     | ✅         |           |

Note: We only list models that we have prepare the [AWQ searching results](https://huggingface.co/datasets/mit-han-lab/awq-model-zoo/tree/main) in the table above. AWQ also supports models such as LLaVA-v1.5 7B, and you may need to run the [AWQ search](#usage) on your own to quantize these models.

## Examples

AWQ can be easily applied to various LMs thanks to its good generalization, including instruction-tuned models and multi-modal LMs. It provides an easy-to-use tool to reduce the serving cost of LLMs.

Here we provide two examples of AWQ application: Vicuna-7B (chatbot) and LLaVA-13B (visual reasoning) under `./examples` directory. AWQ can easily reduce the GPU memory of model serving and speed up token generation. It provides accurate quantization, providing reasoning outputs. You should be able to observe **memory savings** when running the models with 4-bit weights. 

Note that we perform AWQ using only textual calibration data, depsite we are running on multi-modal input. Please refer to `./examples` for details.

![overview](figures/example_vis.jpg)

## Usage

We provide several sample script to run AWQ (please refer to `./scripts`). We use OPT-6.7B as an example.

1. Perform AWQ search and save search results (we already did it for you):
```bash
python -m awq.entry --model_path /PATH/TO/OPT/opt-6.7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/opt-6.7b-w4-g128.pt
```

2. Evaluate the AWQ quantized model on WikiText-2 (simulated pseudo quantization)
```bash
python -m awq.entry --model_path /PATH/TO/OPT/opt-6.7b \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/opt-6.7b-w4-g128.pt \
    --q_backend fake
```

3. Generate real quantized weights (INT4)
```bash
mkdir quant_cache
python -m awq.entry --model_path /PATH/TO/OPT/opt-6.7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/opt-6.7b-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/opt-6.7b-w4-g128-awq.pt
```

4. Load and evaluate the real quantized model (now you can see smaller gpu memory usage)
```bash
python -m awq.entry --model_path /PATH/TO/OPT/opt-6.7b \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/opt-6.7b-w4-g128-awq.pt
```

## Results on Vision-Language Models (VILA-7b/13B)

AWQ also seamlessly supports large multi-modal models (LMMs). We demonstrate the results on the recent [VILA](https://github.com/Efficient-Large-Model/VILA) model family.

| VILA-7B     | VQA-v2            | GQA               | VizWiz  | ScienceQA         | TextVQA           | POPE    | MME     | MMBench           | MMBench-CN    | SEED    |
| ----------- |:-----------------:|:-----------------:|:-------:|:-----------------:|:-----------------:|:-------:|:-------:|:-----------------:|:-------------:|:-------:|
| FP16        | 80.3              | 63.1              | 59.6    | 68.0              | 62.6              | 86.3    | 1489.4  | 69.8              | 61.0          | 61.7    | 
| AWQ-INT4    | 80.1              | 63.0              | 57.8    | 68.3              | 61.9              | 85.3    | 1486.3  | 68.8              | 58.9          | 61.3    |

| VILA-13B    | VQA-v2            | GQA               | VizWiz  | ScienceQA         | TextVQA           | POPE    | MME     | MMBench           | MMBench-CN    | SEED    |
| ----------- |:-----------------:|:-----------------:|:-------:|:-----------------:|:-----------------:|:-------:|:-------:|:-----------------:|:-------------:|:-------:|
| FP16        | 80.5              | 63.6              | 63.1    | 70.5              | 64.0              | 86.3    | 1553.6  | 73.8              | 66.7          | 62.8    | 
| AWQ-INT4    | 80.4              | 63.6              | 63.0    | 71.2              | 63.5              | 87.0    | 1552.9  | 73.6              | 66.3          | 62.2    |


## Inference speed ( Token/sec )

| $~~~~~~$ | Precision |  A100 | 4090 | Orin |
| --- | --- |--- | --- | --- |
| VILA-7B | fp16 | 81.6 | 58.5 | 11.5 |
| VILA-7B-AWQ| int4  |155.3| 168.1| 35.6 |
| VILA-13B | fp16 | 48.5 | OOM | 6.1 |
| VILA-13B-AWQ | int4  | 102.1| 99.0| 17.5 |


## Reference

If you find AWQ useful or relevant to your research, please kindly cite our paper:

```
@inproceedings{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}
```

## Related Projects

[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323)

[Vicuna and FastChat](https://github.com/lm-sys/FastChat#readme)

[LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)

