#!/bin/bash

# ================= 环境变量配置 =================
# SwanLab 配置
export SWANLAB_PROJECT=Qwen3-VL-Video-GRPO-Debug
export SWANLAB_RUN_NAME=qwen3-vl-2b-npu-dist-4card
export SWANLAB_MODE=cloud  # 分布式建议使用 cloud 以便只有主进程上报

# NPU 配置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# 分布式通信后端设置 (昇腾使用 hccl)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 调试配置 (可选)
export CUDA_LAUNCH_BLOCKING=1 

# ================= 路径配置 =================
MODEL_PATH="Qwen3-VL-2B-Thinking" 
DATA_PATH="./dataset/video-mtr/qv-nextgqa_merge_8k.json"
OUTPUT_DIR="./ckpt_dist/$SWANLAB_PROJECT/$SWANLAB_RUN_NAME"
DS_CONFIG="scripts/zero3.json"  # DeepSpeed 配置文件路径

mkdir -p $OUTPUT_DIR

# ================= 启动命令 =================
# 使用 torchrun 进行分布式启动
# --nproc_per_node=4 对应 4 张卡

torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1_video/grpo.py \
    --deepspeed $DS_CONFIG \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --dataset_name video_qa_custom \
    --jsonl_path $DATA_PATH \
    --reward_funcs accuracy format temporal \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 4 \
    --learning_rate 2e-6 \
    --beta 0.04 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --remove_unused_columns false \
    --ddp_timeout 1800