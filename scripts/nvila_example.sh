# run AWQ search
python -m awq.entry --model_path PATH/TO/NVILA \
    --smooth_scale --media_path https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4  \
    --act_scale_path awq_cache/NVILA-VT-smooth-scale.pt --vila-20 \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/NVILA.pt

# generate real quantized weights (w4)
python -m awq.entry --model_path PATH/TO/NVILA/llm \
    --w_bit 4 --q_group_size 128  \
    --load_awq awq_cache/NVILA.pt \
    --q_backend real --dump_quant quant_cache/NVILA-w4-g128-awq.pt --vila-20
