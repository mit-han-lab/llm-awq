MODEL=qwen2.5-7b

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt \
    --dtype bfloat16

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend fake \
    --dtype bfloat16

# generate real quantized weights (w4)
python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w4-g128-awq.pt \
    --dtype bfloat16

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --model_path /dataset/models/llama3/$MODEL \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/$MODEL-w4-g128-awq.pt \
    --dtype bfloat16