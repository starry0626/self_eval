#!/bin/bash

# ================= SDPO 训练启动脚本 =================
# On-Policy Self-Distillation for Video QA
# 基于 Qwen3-VL 模型，使用 Video-R2 数据集进行自蒸馏训练

# ================= 环境变量配置 =================
# SwanLab 实验跟踪配置
export SWANLAB_PROJECT=Qwen3-VL-SDPO-Video
export SWANLAB_RUN_NAME=qwen3-vl-2b-sdpo-video-r2

# 离线模式（可选）
# export SWANLAB_MODE=local

# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# 开启 CUDA 报错堆栈追踪
export CUDA_LAUNCH_BLOCKING=1

# ================= 路径配置 =================
# 模型路径（支持本地路径或 HuggingFace 模型 ID）
MODEL_PATH="Qwen/Qwen3-VL-2B-Thinking"

# 数据集路径
DATA_PATH="./dataset/video-r2/video-r2-grpo-dataset.json"

# 视频文件基础目录（用于视频帧采样）
VIDEO_BASE_DIR="./dataset/video-r2/Video-R2"

# 输出目录
OUTPUT_DIR="./output/sdpo/$SWANLAB_PROJECT/$SWANLAB_RUN_NAME"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# ================= 教师上下文配置 =================
# 控制教师模型额外输入的各部分是否包含
INCLUDE_ANSWER=true              # 包含标准答案
INCLUDE_TEMPORAL_TEXT=true       # 包含时间定位文本
INCLUDE_TEMPORAL_VIDEO=false     # 包含时间定位视频片段（需要视频文件）
INCLUDE_REASONING=true           # 包含推理流程

# 视频采样参数
TEMPORAL_FPS=1.0                 # 时间段视频采样帧率（每秒采样帧数）
TEMPORAL_MAX_FRAMES=8            # 时间段采样的最大帧数（与 fps 冲突时优先限制帧数）
# TEACHER_TEMPORAL_MAX_PIXELS=    # 教师额外视觉输入最大像素数（注释掉则使用处理器默认值）
MAX_TEMPORAL_SEGMENTS=5          # 最大时间片段数量

# ================= 固定教师模型配置 =================
USE_FIXED_TEACHER=false          # 是否使用固定的独立教师模型（参数不随训练更新）
TEACHER_MODEL_PATH=""            # 固定教师模型路径（空则使用与学生相同的模型）

# ================= 散度计算配置 =================
DIVERGENCE_METHOD=full            # 散度计算方法: full, top_k, k3
DIVERGENCE_TOP_K=20              # Top-k 估计的 k 值（仅当 method=top_k 时有效）

# ================= 训练参数配置 =================
# 序列长度
MAX_PROMPT_LENGTH=8192           # 最大 prompt 长度
MAX_COMPLETION_LENGTH=1024       # 最大生成长度

# 学习率和优化
LEARNING_RATE=2e-6               # 学习率
BETA=0.04                        # SDPO 的 beta 参数（暂未使用，保留兼容）
WARMUP_RATIO=0.05                # 预热比例

# 批次大小
PER_DEVICE_TRAIN_BATCH_SIZE=2    # 每设备训练批次大小
GRADIENT_ACCUMULATION_STEPS=1    # 梯度累积步数

# 训练轮次
NUM_TRAIN_EPOCHS=1               # 训练轮次

# 精度和优化
BF16=true                        # 使用 BF16 精度
GRADIENT_CHECKPOINTING=true      # 梯度检查点（节省显存）
ATTN_IMPLEMENTATION=flash_attention_2  # 注意力实现

# 日志和保存
LOGGING_STEPS=1                  # 日志记录步数
SAVE_STEPS=100                   # 模型保存步数
SAVE_ONLY_MODEL=true             # 只保存模型权重

# ================= 启动命令 =================
# 构建可选参数
OPTIONAL_ARGS=""
if [ -n "$TEACHER_TEMPORAL_MAX_PIXELS" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --teacher_temporal_max_pixels $TEACHER_TEMPORAL_MAX_PIXELS"
fi
if [ -n "$TEACHER_MODEL_PATH" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --teacher_model_path $TEACHER_MODEL_PATH"
fi

python src/open_r1_video/sdpo.py \
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
