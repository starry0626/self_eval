#!/bin/bash

# ================= 环境变量配置 =================
# --- 修改点 1: 替换 WANDB 环境变量为 SWANLAB ---
export SWANLAB_PROJECT=Qwen3-VL-Video-GRPO-Debug
export SWANLAB_RUN_NAME=qwen3-vl-2b-npu-debug
# 对应 WANDB_MODE=offline，SwanLab 使用 cloud (云端) 或 local (本地离线)
# 如果你想离线调试，使用 "local"；如果想同步到网页端，使用 "cloud"
# export SWANLAB_MODE=local 

# 指定使用 0 号 NPU 卡
export ASCEND_RT_VISIBLE_DEVICES=0
# 开启 CUDA(NPU) 报错堆栈追踪
export CUDA_LAUNCH_BLOCKING=1 

# ================= 路径配置 =================
MODEL_PATH="Qwen3-VL-2B-Thinking" 
DATA_PATH="./dataset/video-mtr/qv-nextgqa_merge_8k.json"
OUTPUT_DIR="./ckpt_debug/$SWANLAB_PROJECT/$SWANLAB_RUN_NAME"

mkdir -p $OUTPUT_DIR

# ================= 启动命令 =================
python src/open_r1_video/grpo.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name video_qa_custom \
    --jsonl_path $DATA_PATH \
    --reward_funcs accuracy format temporal \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 2 \
    --learning_rate 2e-6 \
    --beta 0.04 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to swanlab \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $SWANLAB_RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --warmup_ratio 0.05 \
    --dataloader_pin_memory false \
    --remove_unused_columns false