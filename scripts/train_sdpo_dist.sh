#!/bin/bash

# ================= SDPO 分布式训练启动脚本 =================
# On-Policy Self-Distillation for Video QA
# 单机多卡 · torchrun 启动 · DeepSpeed ZeRO-3

# ================= GPU 配置 =================
# 参与训练的 GPU 编号（逗号分隔）
CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES

# 自动统计 GPU 数量（根据上面的 CUDA_VISIBLE_DEVICES 推断）
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# torchrun 通信配置
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500                # 如果端口被占用，修改此值

# ================= DeepSpeed 配置 =================
# zero3.json:         纯 GPU，显存充足时推荐（速度更快）
# zero3_offload.json: 优化器 + 参数卸载到 CPU，显存不足时使用（速度略慢）
DEEPSPEED_CONFIG="scripts/zero3.json"

# ================= 环境变量配置 =================
export SWANLAB_PROJECT=Qwen3-VL-SDPO-Video
export SWANLAB_RUN_NAME=qwen3-vl-2b-sdpo-video-r2-dist

# 离线模式（可选）
# export SWANLAB_MODE=local

# 注意：分布式训练中不要开启 CUDA_LAUNCH_BLOCKING，会强制序列化所有 CUDA 调用导致严重性能下降
# export CUDA_LAUNCH_BLOCKING=1  # 仅调试时开启

# NCCL 通信优化（可选）
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1       # 无 InfiniBand 时可开启

# ================= 路径配置 =================
MODEL_PATH="./Qwen3-VL-2B-Thinking"
DATA_PATH="./dataset/video-r2/Video-R2/video-r2-grpo-dataset.json"
VIDEO_BASE_DIR="./video_data/videos"
OUTPUT_DIR="./output/sdpo/$SWANLAB_PROJECT/$SWANLAB_RUN_NAME"

mkdir -p $OUTPUT_DIR

# ================= 教师上下文配置 =================
INCLUDE_ANSWER=true
INCLUDE_TEMPORAL_TEXT=true
INCLUDE_TEMPORAL_VIDEO=true
INCLUDE_REASONING=true

TEMPORAL_FPS=1.0
TEMPORAL_MAX_FRAMES=8
# TEACHER_TEMPORAL_MAX_PIXELS=    # 注释掉则使用处理器默认值
MAX_TEMPORAL_SEGMENTS=5

# ================= 固定教师模型配置 =================
# 注意：ZeRO-3 下固定教师模型的参数也会被分片，USE_FIXED_TEACHER=true 时请确认显存足够
USE_FIXED_TEACHER=false
TEACHER_MODEL_PATH=""

# ================= 散度计算配置 =================
DIVERGENCE_METHOD=full
DIVERGENCE_TOP_K=20

# ================= 数据集配置 =================
# MAX_TRAIN_SAMPLES=

# ================= 训练参数配置 =================
MAX_PROMPT_LENGTH=16384
MAX_COMPLETION_LENGTH=1024

LEARNING_RATE=2e-6
BETA=0.04
WARMUP_RATIO=0.05

# 每卡 batch size；总有效 batch = NUM_GPUS × PER_DEVICE_TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1

NUM_TRAIN_EPOCHS=1

BF16=true
GRADIENT_CHECKPOINTING=true
ATTN_IMPLEMENTATION=flash_attention_2

LOGGING_STEPS=1
SAVE_STEPS=100
SAVE_ONLY_MODEL=true

# ================= 构建可选参数 =================
OPTIONAL_ARGS=""
if [ -n "$TEACHER_TEMPORAL_MAX_PIXELS" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --teacher_temporal_max_pixels $TEACHER_TEMPORAL_MAX_PIXELS"
fi
if [ -n "$TEACHER_MODEL_PATH" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --teacher_model_path $TEACHER_MODEL_PATH"
fi
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --max_train_samples $MAX_TRAIN_SAMPLES"
fi

# ================= 打印启动信息 =================
echo "=========================================="
echo "  SDPO 分布式训练"
echo "  GPU:        $CUDA_VISIBLE_DEVICES ($NUM_GPUS 卡)"
echo "  DeepSpeed:  $DEEPSPEED_CONFIG"
echo "  模型:       $MODEL_PATH"
echo "  输出:       $OUTPUT_DIR"
echo "=========================================="

# ================= 启动命令 =================
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1_video/sdpo.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH \
    --jsonl_path $DATA_PATH \
    --video_base_dir $VIDEO_BASE_DIR \
    \
    `# 教师上下文配置` \
    --include_answer $INCLUDE_ANSWER \
    --include_temporal_text $INCLUDE_TEMPORAL_TEXT \
    --include_temporal_video $INCLUDE_TEMPORAL_VIDEO \
    --include_reasoning $INCLUDE_REASONING \
    --temporal_fps $TEMPORAL_FPS \
    --temporal_max_frames $TEMPORAL_MAX_FRAMES \
    --max_temporal_segments $MAX_TEMPORAL_SEGMENTS \
    \
    `# 固定教师模型配置` \
    --use_fixed_teacher $USE_FIXED_TEACHER \
    \
    `# 散度计算配置` \
    --divergence_method $DIVERGENCE_METHOD \
    --divergence_top_k $DIVERGENCE_TOP_K \
    \
    `# 序列长度` \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    \
    `# 学习率和优化` \
    --learning_rate $LEARNING_RATE \
    --beta $BETA \
    --warmup_ratio $WARMUP_RATIO \
    \
    `# 批次大小` \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    \
    `# 训练轮次` \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    \
    `# 精度和优化` \
    --bf16 $BF16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --attn_implementation $ATTN_IMPLEMENTATION \
    \
    `# 日志和保存` \
    --logging_steps $LOGGING_STEPS \
    --report_to swanlab \
    --run_name $SWANLAB_RUN_NAME \
    --save_steps $SAVE_STEPS \
    --save_only_model $SAVE_ONLY_MODEL \
    \
    `# 其他` \
    --data_seed 42 \
    --dataloader_pin_memory false \
    --remove_unused_columns false \
    $OPTIONAL_ARGS

echo "Training completed! Output saved to: $OUTPUT_DIR"
