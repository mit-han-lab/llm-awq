MODEL=llava-13b-v0

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path /dataset/llava-hf/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# generate real quantized weights (w4)
python -m awq.entry --model_path /dataset/llava-hf/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w4-g128-awq.pt
