MODEL_PATH=PATH_TO_NVILA
MODEL_NAME=NVILA-8B

# run AWQ search
python -m awq.entry --model_path $MODEL_PATH \
    --smooth_scale --media_path https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4  \
    --act_scale_path awq_cache/$MODEL_NAME-smooth-scale.pt --vila-20 \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL_NAME.pt

# generate real quantized weights (w4)
python -m awq.entry --model_path $MODEL_PATH/llm \
    --w_bit 4 --q_group_size 128  \
    --load_awq awq_cache/$MODEL_NAME.pt \
    --q_backend real --dump_quant quant_cache/$MODEL_NAME-w4-g128-awq.pt --vila-20

# Run the TinyChat demo:
python nvila_demo.py --model-path $MODEL_PATH    \
       --quant_path quant_cache/$MODEL_NAME-w4-g128-awq.pt      \
       --media ../figures/nvila-logo.jpg     \
       --act_scale_path awq_cache/$MODEL_NAME-smooth-scale.pt    \
       --all --chunk --model_type nvila --vis_image