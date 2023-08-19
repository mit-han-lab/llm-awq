# TinyChat: Efficient and Lightweight Chatbot with AWQ

We introduce TinyChat, a cutting-edge chatbot interface designed for lightweight resource consumption and fast inference speed on GPU platforms. It allows for seamless deployment on consumer-level GPUs such as 3090/4090 and low-power edge devices like the NVIDIA Jetson Orin, empowering users with a responsive conversational experience like never before.



The current release supports:

- LLaMA-2-7B/13B-chat;
- Vicuna;
- MPT-chat;
- Falcon-instruct.


## Contents

- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Usage](#usage)
- [Reference](#reference)


## Examples

Thanks to AWQ, TinyChat can now deliver more prompt responses through 4-bit inference. The following examples showcase that TinyChat's W4A16 generation is 2.3x faster on RTX 4090 and 1.4x faster on Jetson Orin, compared to the FP16 baselines. (Tested with [LLaMA-2-7b]( https://huggingface.co/meta-llama/Llama-2-7b-chat-hf ) model.)

* TinyChat on RTX 4090:

![TinyChat on RTX 4090: W4A16 is 2.3x faster than FP16](./figures/4090_example.gif)

* TinyChat on Jetson Orin:

![TinyChat on Jetson Orin: W4A16 is 1.4x faster than FP16](./figures/orin_example.gif)


## Benchmarks

We benchmark TinyChat on A6000 (server-class GPU), 4090 (desktop GPU) and Orin (edge GPU).

We use the default implementation from Huggingface for the FP16 baseline. The INT4 implementation applies AWQ and utilizes our fast W4A16 GPU kernel. Please notice that the end-to-end runtime for INT4 TinyChat could be further improved if we reduce the framework overhead from Huggingface (e.g. utilizing the implementation from TGI). We are working on a new release with even faster inference performance, please stay tuned!

The latency reported in all tables are per-token latency for the generation stage.

### A6000 Results

| Model       | FP16 latency (ms) | INT4 latency (ms) | Speedup |
| ----------- |:-----------------:|:-----------------:|:-------:|
| LLaMA-2-7B  | 27.14             | 12.44             | 2.18x   |
| LLaMA-2-13B | 47.28             | 20.28             | 2.33x   |
| Vicuna-7B   | 26.06             | 12.43             | 2.10x   |
| Vicuna-13B  | 44.91             | 17.30             | 2.60x   |
| MPT-7B      | 22.79             | 16.87             | 1.35x   |
| MPT-30B     | OOM               | 31.57             | --      |
| Falcon-7B   | 39.44             | 27.34             | 1.44x   |

### 4090 Results

| Model       | FP16 latency (ms) | INT4 latency (ms) | Speedup |
| ----------- |:-----------------:|:-----------------:|:-------:|
| LLaMA-2-7B  | 19.97             | 8.66              | 2.31x   |
| LLaMA-2-13B | OOM               | 13.54             | --      |
| Vicuna-7B   | 19.09             | 8.61              | 2.22x   |
| Vicuna-13B  | OOM               | 12.17             | --      |
| MPT-7B      | 17.09             | 12.58             | 1.36x   |
| MPT-30B     | OOM               | 23.54             | --      |
| Falcon-7B   | 29.91             | 19.84             | 1.51x   |

### Orin Results

| Model       | FP16 latency (ms) | INT4 latency (ms) | Speedup |
| ----------- |:-----------------:|:-----------------:|:-------:|
| LLaMA-2-7B  | 104.71            | 75.11             | 1.39x   |
| LLaMA-2-13B | OOM               | 136.81            | --      |
| Vicuna-7B   | 93.12             | 65.34             | 1.43x   |
| Vicuna-13B  | OOM               | 115.4             | --      |
| MPT-7B      | 89.85             | 67.36             | 1.33x   |
| Falcon-7B   | 147.84            | 102.74            | 1.44x   |


## Usage

1. Please follow the [AWQ installation guidance](https://github.com/mit-han-lab/llm-awq#readme) to install AWQ and its dependencies.

2. Download the pretrained instruction-tuned LLMs:
   
   - For LLaMA-2-chat, please refer to [this link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf);
   - For Vicuna, please refer to [this link](https://huggingface.co/lmsys/);
   - For MPT-chat, please refer to [this link](https://huggingface.co/mosaicml/mpt-7b-chat);
   - For Falcon-instruct, please refer to [this link](https://huggingface.co/tiiuae/falcon-7b-instruct).

3. Quantize instruction-tuned LLMs with AWQ (see [usage in README](../README.md#usage)).

4. Run the TinyChat demo:

Here, we use Vicuna as an example and assume that you have already quantized the model.

```bash
cd tinychat
python demo.py --model_path vicuna-7b-v1.5-awq
```

You may also run the following command to execute the chatbot in FP16 to compare the speed and quality of language generation:

```bash
python demo.py --model_path lmsys/vicuna-7b-v1.5 --precision W16A16
```

## Reference

TinyChat is inspired by the following open-source projects: [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), [vLLM](https://github.com/vllm-project/vllm), [FastChat](https://github.com/lm-sys/FastChat).
