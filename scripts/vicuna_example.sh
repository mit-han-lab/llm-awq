MODEL=lmsys/vicuna-7b-v1.5

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --entry_type search \
    --model_path $MODEL \
    --search_path $MODEL-awq

# generate real quantized weights (w4)
python -m awq.entry --entry_type quant \
    --model_path $MODEL \
    --search_path $MODEL-awq/awq_model_search_result.pt \
    --quant_path $MODEL-awq

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --entry_type perplexity \
    --quant_path $MODEL-awq \
    --quant_file awq_model_w4_g128.pt