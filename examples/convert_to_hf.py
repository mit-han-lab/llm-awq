# This script demonstrates how you can convert your model into HF format
# easily and push the quantized weights on the Hub using simple tools.
# Make sure to have transformers > 4.34 and that you have ran 
# `huggingface-cli login` on your terminal before running this 
# script
import os
import argparse

# This demo only support single GPU for now
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoConfig, AWQConfig, AutoTokenizer
from huggingface_hub import HfApi

api = HfApi()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of the original hf model', required=True)
parser.add_argument("--quantized_model_path", type=str, help='path of the quantized AWQ model', required=True)
parser.add_argument("--quantized_model_hub_path", type=str, help='path of the quantized AWQ model to push on the Hub', required=True)
parser.add_argument('--w_bit', type=int, default=4, help='')
parser.add_argument("--q_group_size", default=128, type=int)
parser.add_argument("--no_zero_point", action='store_true')

args = parser.parse_args()

original_model_path = args.model_path
quantized_model_path = args.quantized_model_path
quantized_model_hub_path = args.quantized_model_hub_path

# Load the corresponding AWQConfig
quantization_config = AWQConfig(
    w_bit=args.w_bit,
    q_group_size=args.q_group_size,
    zero_point=not args.no_zero_point,
    backend="llm-awq",
    version="GEMV",
)

# Set the attribute `quantization_config` in model's config
config = AutoConfig.from_pretrained(original_model_path)
config.quantization_config = quantization_config

# Load tokenizer
tok = AutoTokenizer.from_pretrained(original_model_path)

# Push config and tokenizer
config.push_to_hub(quantized_model_hub_path)
tok.push_to_hub(quantized_model_hub_path)

# Upload model weights
api.upload_file(
    path_or_fileobj=quantized_model_path,
    path_in_repo="pytorch_model.bin",
    repo_id=quantized_model_hub_path,
    repo_type="model",
)
