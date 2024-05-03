MODEL=VILA1.5-7b

# run AWQ search (optional; we provided the pre-computed results)
# Note: vila checkpoints are now stored in 3 parts.
# only llm folder will be quantized
python -m awq.entry --model_path /dataset/vila-hf/$MODEL/llm \
    --w_bit 4 --q_group_size 128 --vila-15 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# generate real quantized weights (w4)
python -m awq.entry --model_path /dataset/vila-hf/$MODEL/llm \
    --w_bit 4 --q_group_size 128 --vila-15 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant /dataset/vila-hf/$MODEL-awq/llm/$MODEL-w4-g128-awq.pt