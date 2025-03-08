MODEL_NAME="NVILA-8B"
MODEL_PATH="/home/yuming/workspace/models/NVILA-8B"
# AWQ_PATH="../llm-awq/awq_cache/${MODEL_NAME}-w4-g128.pt"
AWQ_PATH="awq_cache/${MODEL_NAME}-w4-g128.pt"
QUANT_PATH="quant_cache/${MODEL_NAME}-w4-g128-awq.pt"
MEDIA_PATH="/home/yuming/workspace/tinychat2nvila/awq_kf/figures/vila-logo.jpg"
LLM_PATH="${MODEL_PATH}/llm"

python -m awq.entry --model_path $MODEL_PATH \
    --smooth_scale --media_path https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4  \
    --act_scale_path awq_cache/NVILA-8B-VT-smooth-scale.pt --vila-20 \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq $AWQ_PATH

python -m awq.entry --model_path $LLM_PATH \
    --w_bit 4 --q_group_size 128  \
    --load_awq $AWQ_PATH \
    --q_backend real --dump_quant $QUANT_PATH --vila-20
cd tinychat
python nvila_demo.py --model-path $MODEL_PATH       \
    --quant_path $QUANT_PATH      \
    --media $MEDIA_PATH    \
    --act_scale_path ../awq_cache/NVILA-8B-VT-smooth-scale.pt \
    --all --chunk --model_type nvila

python nvila_demo.py --model-path /home/yuming/workspace/models/NVILA-8B       \
    --quant_path ../quant_cache/NVILA-8B-w4-g128-awq-v2.pt      \
    --media /home/yuming/workspace/tinychat2nvila/awq_kf/figures/vila-logo.jpg    \
    --act_scale_path ../awq_cache/NVILA-8B-VT-smooth-scale.pt \
    --all --chunk --model_type nvila