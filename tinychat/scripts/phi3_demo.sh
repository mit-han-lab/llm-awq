MODEL_PATH=/data/lujunli/hf_download/
MODEL_NAME=Phi-3-mini-128k-instruct

export CUDA_VISIBLE_DEVICES=1,2

# # Perform AWQ search and save search results (we already did it for you):
# mkdir -p awq_cache
# python -m awq.entry --model_path $MODEL_PATH/$MODEL_NAME \
#     --w_bit 4 --q_group_size 128 \
#     --run_awq --dump_awq awq_cache/phi-3-chat-w4-g128.pt

# # Generate real quantized weights (INT4):
# mkdir -p quant_cache
# python -m awq.entry --model_path $MODEL_PATH/$MODEL_NAME \
#     --w_bit 4 --q_group_size 128 \
#     --load_awq awq_cache/phi-3-chat-w4-g128.pt \
#     --q_backend real --dump_quant quant_cache/phi-3-chat-w4-g128-awq.pt

# # Run the TinyChat demo:

CUDA_VISIBLE_DEVICES=1 python demo.py --model_type phi3 \
    --model_path $MODEL_PATH/$MODEL_NAME \
    --q_group_size 128 --load_quant quant_cache/phi-3-chat-w4-g128-awq-v2.pt \
    --precision W4A16

# # Split checkpoint into shards for mem-efficient loading:
# python split_ckpt.py --input_path quant_cache/phi-3-chat-w4-g128-awq.pt \
#     --output_path quant_cache/phi-3-chat-w4-g128-awq

# # Run the TinyChat demo in mem_efficient_load mode:
# python demo.py --model_type llama \
#     --model_path $MODEL_PATH/$MODEL_NAME \
#     --q_group_size 128 --load_quant quant_cache/phi-3-chat-w4-g128-awq \
#     --precision W4A16 --mem_efficient_load
