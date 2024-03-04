# TinyChat: Efficient and Lightweight Chatbot with AWQ

We introduce TinyChat, a cutting-edge chatbot interface designed for lightweight resource consumption and fast inference speed on GPU platforms. It allows for seamless deployment on consumer-level GPUs such as 3090/4090 and low-power edge devices like the NVIDIA Jetson Orin, empowering users with a responsive conversational experience like never before.

The current release supports:

- VILA-7B/13B;

- LLaVA-7B/13B;

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

**Thanks to AWQ, TinyChat can now deliver more prompt responses through 4-bit inference. The following examples showcase that TinyChat's W4A16 generation is up to 3.7x faster on RTX 4090 and 3.3x faster on Jetson Orin, compared to the FP16 baselines. (Tested with [LLaMA-2-7b]( https://huggingface.co/meta-llama/Llama-2-7b-chat-hf ) model.)**

* TinyChat on RTX 4090 (3.4x faster than FP16):

![TinyChat on RTX 4090: W4A16 is 3.4x faster than FP16](./figures/4090_example.gif)

* TinyChat on Jetson Orin (3.2x faster than FP16):

![TinyChat on Jetson Orin: W4A16 is 3.2x faster than FP16](./figures/orin_example.gif)

**TinyChat also supports inference with vision language models (e.g., VILA, LLaVA). In the following examples, W4A16 quantized models from VILA family are launched with TinyChat.**

* TinyChat with VILA-13B on RTX 4090 (multi-image inputs supported):

![TinyChat with VILA on 4090](./figures/4090_vila_example.gif)

* TinyChat with VILA-7B/13B on Jetson Orin:

![TinyChat with VILA on Orin](./figures/orin_vila_example.gif)


## Benchmarks

We benchmark TinyChat on A6000 (server-class GPU), 4090 (desktop GPU) and Orin (edge GPU).

We use the default implementation from Huggingface for the FP16 baseline. The INT4 implementation applies AWQ and utilizes our fast W4A16 GPU kernel. We also apply additional optimization techniques in the latest release. For example, we fuse all the operations in MHA/GQA/MQA into a single kernel, and fuse positional embedding kernels into the attention kernel. We also pre-allocate key-value caches to avoid the online memory allocation overhead from Huggingface.

The latency reported in all tables are per-token latency for the generation stage.

### A6000 Results

| Model       | FP16 latency (ms) | INT4 latency (ms) | Speedup |
| ----------- |:-----------------:|:-----------------:|:-------:|
| LLaMA-2-7B  | 27.14             | 8.71              | 3.12x   |
| LLaMA-2-13B | 47.28             | 14.64             | 3.23x   |
| Vicuna-7B   | 26.06             | 8.39              | 3.11x   | 
| Vicuna-13B  | 44.91             | 13.46             | 3.34x   |
| MPT-7B      | 22.79             | 7.99              | 2.85x   |
| MPT-30B     | OOM               | 28.15             | --      |  
| Falcon-7B   | 39.44             | 11.71             | 3.37x   |
| VILA-7B     | 23.60             | 8.14              | 2.90x   |
| VILA-13B    | 46.58             | 13.74             | 3.39x   |

### 4090 Results

| Model       | FP16 latency (ms) | INT4 latency (ms) | Speedup |
| ----------- |:-----------------:|:-----------------:|:-------:|
| LLaMA-2-7B  | 19.97             | 6.02*             | 3.31x   |
| LLaMA-2-13B | OOM               | 10.35             | --      |
| Vicuna-7B   | 19.09             | 5.33              | 3.58x   |
| Vicuna-13B  | OOM               | 9.17              | --      |
| MPT-7B      | 17.09             | 6.18              | 2.77x   |
| MPT-30B     | OOM               | 20.60             | --      |
| Falcon-7B   | 29.91             | 8.02              | 3.73x   |
| VILA-7B     | 17.09             | 5.95              | 2.87x   |
| VILA-13B    | OOM               | 10.01             | --      |

*: The reason why LLaMA-2-7B is slower than Vicuna-7B is because we need a longer prompt (with > 500 tokens) to prevent the model from talking with itself. If we use the benchmarking strategy from exLLaMA (i.e. only 4 context tokens), our speed is around 195 tokens / second.

### Orin Results

| Model       | FP16 latency (ms) | INT4 latency (ms) | Speedup |
| ----------- |:-----------------:|:-----------------:|:-------:|
| LLaMA-2-7B  | 104.71            | 33.07*            | 3.17x   | 
| LLaMA-2-13B | OOM               | 58.20             | --      |
| Vicuna-7B   | 93.12             | 30.73             | 3.03x   |
| Vicuna-13B  | OOM               | 54.98             | --      |
| MPT-7B      | 89.85             | 31.22             | 2.88x   |
| Falcon-7B   | 147.84            | 45.10             | 3.28x   |
| VILA-7B     | 86.95             | 28.09             | 3.10x   |
| VILA-13B    | OOM               | 57.14             | --      |

*: We can similarly achieve 33 tokens / second on Orin if we use the benchmarking strategy from exLLaMA.

## Evaluation

We recently evaluated AWQ's performance on tVision Language Models. Here is a summary of VILA results.

| VILA-7B     | VQA-v2            | GQA               | VizWiz  | ScienceQA         | TextVQA           | POPE    | MME     | MMBench           | MMBench-CN    | SEED    |
| ----------- |:-----------------:|:-----------------:|:-------:|:-----------------:|:-----------------:|:-------:|:-------:|:-----------------:|:-------------:|:-------:|
| FP16        | 80.3              | 63.1              | 59.6    | 68.0              | 62.6              | 86.3    | 1489.4  | 69.8              | 61.0          | 61.7    | 
| AWQ-INT4    | 80.1              | 63.0              | 57.8    | 68.3              | 61.9              | 85.3    | 1486.3  | 68.8              | 58.9          | 61.3    |

| VILA-13B    | VQA-v2            | GQA               | VizWiz  | ScienceQA         | TextVQA           | POPE    | MME     | MMBench           | MMBench-CN    | SEED    |
| ----------- |:-----------------:|:-----------------:|:-------:|:-----------------:|:-----------------:|:-------:|:-------:|:-----------------:|:-------------:|:-------:|
| FP16        | 80.5              | 63.6              | 63.1    | 70.5              | 64.0              | 86.3    | 1553.6  | 73.8              | 66.7          | 62.8    | 
| AWQ-INT4    | 80.4              | 63.6              | 63.0    | 71.2              | 63.5              | 87.0    | 1552.9  | 73.6              | 66.3          | 62.2    |

## Usage

1. Please follow the [AWQ installation guidance](https://github.com/mit-han-lab/llm-awq#readme) to install AWQ and its dependencies.

2. Download the pretrained instruction-tuned LLMs:
   
   - For LLaMA-2-chat, please refer to [this link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf);
   
   - For Vicuna, please refer to [this link](https://huggingface.co/lmsys/);
   
   - For MPT-chat, please refer to [this link](https://huggingface.co/mosaicml/mpt-7b-chat);
   
   - For Falcon-instruct, please refer to [this link](https://huggingface.co/tiiuae/falcon-7b-instruct).

3. Quantize instruction-tuned LLMs with AWQ:
- We provide pre-computed AWQ search results for multiple model families, including LLaMA, OPT, Vicuna, VILA, and LLaVA. To get the pre-computed AWQ search results, run:

```bash
# git lfs install  # install git lfs if not already
git clone https://huggingface.co/datasets/mit-han-lab/awq-model-zoo awq_cache
```

- You may run a one-line starter below:

```bash
./scripts/llama2_demo.sh
```

Alternatively, you may go through the process step by step. We will demonstrate the quantization process with LLaMA-2. For all other models except Falcon, one only needs to change the `model_path` and saving locations. For Falcon-7B, we also need to change `q_group_size` from 128 to 64.

- Perform AWQ search and save search results (we already did it for you):

```bash
mkdir awq_cache
python -m awq.entry --model_path /PATH/TO/LLAMA2/llama-2-7b-chat \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama-2-7b-chat-w4-g128.pt
```

- Generate real quantized weights (INT4):

```bash
mkdir quant_cache
python -m awq.entry --model_path /PATH/TO/LLAMA2/llama-2-7b-chat \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama-2-7b-chat-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/llama-2-7b-chat-w4-g128-awq.pt
```

4. Run the TinyChat demo:

```bash
cd tinychat
python demo.py --model_type llama \
    --model_path /PATH/TO/LLAMA2/llama-2-7b-chat \
    --q_group_size 128 --load_quant quant_cache/llama-2-7b-chat-w4-g128-awq.pt \ 
    --precision W4A16
```

Note: if you use Falcon-7B-instruct, please remember to also change `q_group_size` to 64. You may also run the following command to execute the chatbot in FP16 to compare the speed and quality of language generation:

```bash
python demo.py --model_type llama \
    --model_path /PATH/TO/LLAMA2/llama-2-7b-chat \
    --precision W16A16
```

The above command works well for most cloud and desktop GPUs, since their CPU and GPU memory space are separated. However, for edge GPUs with shared host and device memory, in order to run larger models (e.g. LLaMA-2-70B on 64GB Orin), it is necessary to break down the pretrained checkpoints into small pieces:

```bash
python split_ckpt.py --input_path quant_cache/llama-2-7b-chat-w4-g128-awq.pt \
    --output_path quant_cache/llama-2-7b-chat-w4-g128-awq
```

Then, to run the demo, one can use the following command. The only changes compared with the demo command above are: 

- We modify the `load_quant` argument;

- We introduce another flag `mem_efficient_load`.

```bash
cd tinychat
python demo.py --model_type llama \
    --model_path /PATH/TO/LLAMA2/llama-2-7b-chat \
    --q_group_size 128 --load_quant quant_cache/llama-2-7b-chat-w4-g128-awq \ 
    --precision W4A16 --mem_efficient_load
```

5. (Optional) Run the benchmark script:

```bash
cd tinychat
python benchmark.py --model_type llama \
    --model_path /PATH/TO/LLAMA2/llama-2-7b-chat    \
    --q_group_size 128
```

Note: The kv caches in the current implementation are pre-allocated. So if you run out of memory, it might be the case that the kv cache is too large. To solve the problem, you may pass in `--max_seq_len [a smaller number]`.

### Support VLM models (VILA & LLaVA)

Our TinyChat also supports vision language models. Follow the instructions below to run VLMs on your own devices!

Step 1-3 are same as the deployment for Language-only models.

1. Follow the [AWQ installation guidance](https://github.com/mit-han-lab/llm-awq#readme) to install AWQ and its dependencies.

2. Download the pretrained VLMs (VILA or LLaVA).

3. Quantize the VLMs with AWQ and get the quantized checkpoint in `quant_cache`. We also provide a [sample script](../scripts/vila_example.sh) for this step.

4. Run the TinyChat demo for VLMs (with vlm_demo.py):

```bash
cd tinychat
python vlm_demo.py \
    --model-path /PATH/TO/VILA/VILA-13B \
    --quant-path quant_cache/vila-13b-w4-g128-awq.pt \ 
    --precision W4A16 \
    --image-file /PATH/TO/INPUT/IMAGE \
    --vis-image #Optional
```

Note: if you enable `--vis-image` mode, TinyChat will print input images directly in your terminal. You may need to install [termvisage](https://github.com/AnonymouX47/termvisage) to enable this mode. A [terminal emulator](https://github.com/AnonymouX47/termvisage?tab=readme-ov-file#requirements) is also required.

Note: VILA model family supports multi-image inputs. You can input multiple images in `/PATH/TO/INPUT/IMAGE` above, each image should be seperated by `,`.


## Reference

TinyChat is inspired by the following open-source projects: [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [vLLM](https://github.com/vllm-project/vllm), [FastChat](https://github.com/lm-sys/FastChat), [llama_cu_awq](https://github.com/ankan-ban/llama_cu_awq), [LLaVA](https://github.com/haotian-liu/LLaVA), [termvisage](https://github.com/AnonymouX47/termvisage).

