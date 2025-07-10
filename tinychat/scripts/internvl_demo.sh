MODEL_PATH=PATH_TO_INTERNVL
MODEL_NAME=InternVL3-8B

# run AWQ search
python -m awq.entry --model_path $MODEL_PATH \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL_NAME-w4-g128.pt

# generate real quantized weights (w4)
python -m awq.entry --model_path $MODEL_PATH \
    --w_bit 4 --q_group_size 128 --load_awq awq_cache/$MODEL_NAME-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL_NAME-w4-128-awq.pt

# Run the TinyChat demo:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python internvl_demo.py --model-path $MODEL_PATH \
    --quant_path quant_cache/$MODEL_NAME-w4-128-awq-v2.pt \
    --media ../figures/vila-logo.jpg --max_seq_len 4096 --chunk \
    --model_type internvl3 --quant_VT --quant_llm