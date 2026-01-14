export VLLM_USE_MODELSCOPE=true

# 1) 选卡（可省略，默认全卡）
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export VLLM_LOGGING_LEVEL=DEBUG


# 2) 稳定性与日志
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_FLASH_ATTENTION=0
export VLLM_USE_TRITON_FLASH_ATTN=0

CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve /data/deepwriting/model/huggingface \
    --port 8001 \
    --gpu-memory-utilization 0.8 \
    --served-model-name qwen3-8b-sft \
    --tensor-parallel-size 4