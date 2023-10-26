# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration 
[[Paper](https://arxiv.org/abs/2306.00978)][[Slides](https://www.dropbox.com/scl/fi/dtnp6h6y1mnp7g036axu6/AWQ-slide.pdf?rlkey=ffgh50hxhx8dmsnjiu8kef0ou&dl=0)][[Video](https://youtu.be/U0yvqjdMfr0)]

**Efficient and accurate** low-bit weight quantization (INT3/4) for LLMs, supporting **instruction-tuned** models and **multi-modal** LMs.

![overview](figures/overview.png)

The current release supports: 

- AWQ search for accurate quantization. 
- Pre-computed AWQ model zoo for LLMs (LLaMA-1&2, OPT, Vicuna, LLaVA; load to generate quantized weights).
- Memory-efficient 4-bit Linear in PyTorch.
- Efficient CUDA kernel implementation for fast inference (support context and decoding stage).
- Examples on 4-bit inference of an instruction-tuned model (Vicuna) and multi-modal LM (LLaVA).

![TinyChat on Orin: W4A16 is 3.2x faster than FP16](./tinychat/figures/orin_example.gif)

Check out [TinyChat](tinychat), which delievers **30 tokens/second** inference performance (**3.2x faster** than FP16) for the **LLaMA-2** chatbot on the resource-constrained NVIDIA Jetson Orin! 

It also offers a turn-key solution for **on-device inference** of LLMs on **resource-constrained edge platforms**. With TinyChat, it is now possible to run **large** models on **small** and **low-power** devices even without Internet connection.


## News
- [2023/11] 🔥 AWQ is now integrated natively in Hugging Face transformers through `from_pretrained`. You can either load quantized models from the Hub or your own HF quantized models.
- [2023/10] AWQ is integrated into NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/)
- [2023/09] AWQ is integrated into [FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/awq.md), [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/quantization_utils/awq.py), [HuggingFace TGI](https://github.com/huggingface/text-generation-inference/pull/1054), and [LMDeploy](https://github.com/InternLM/lmdeploy). 
- [2023/09] ⚡ Check out our latest [**TinyChat**](tinychat), which is ~2x faster than the first release on Orin!
- [2023/09] ⚡ Check out [**AutoAWQ**](https://github.com/casper-hansen/AutoAWQ), a third-party implementation to make AWQ easier to expand to new models, improve inference speed, and integrate into Huggingface.
- [2023/07] 🔥 We released **TinyChat**, an efficient and lightweight chatbot interface based on AWQ. TinyChat enables efficient LLM inference on both cloud and edge GPUs. LLama-2-chat models are supported! Check out our implementation [here](tinychat).
- [2023/07] 🔥 We added AWQ support and pre-computed search results for Llama-2 models (7B & 13B). Checkout our model zoo [here](https://huggingface.co/datasets/mit-han-lab/awq-model-zoo)!
- [2023/07] We extended the support for more LLM models including MPT, Falcon, and BLOOM. 

## Contents

- [Install](#install)
- [AWQ Model Zoo](#awq-model-zoo)
- [Examples](#examples)
- [Usage](#usage)
- [Reference](#reference)

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
| LLaMA-2  | 7B/7B-chat/13B/13B-chat   | ✅         | ✅        |
| LLaMA  | 7B/13B/30B/65B              | ✅         | ✅        |
| OPT    | 125m/1.3B/2.7B/6.7B/13B/30B | ✅         | ✅        |
| Vicuna-v1.1 | 7B/13B                 | ✅         |           |
| LLaVA-v0 | 13B                       | ✅         |           |

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

## Reference

If you find AWQ useful or relevant to your research, please kindly cite our paper:

```
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
```

## Related Projects

[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323)

[Vicuna and FastChat](https://github.com/lm-sys/FastChat#readme)

[LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)

